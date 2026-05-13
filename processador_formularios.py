import os
import sys
import re
import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
import pypdfium2 as pdfium
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import threading

# --- CONFIGURAÇÕES GLOBAIS ---

# --- CONFIGURAÇÃO INTELIGENTE DO TESSERACT PARA O .EXE ---
def obter_caminho_tesseract():
    """
    Descobre se o programa está rodando como script Python normal ou como um .exe compilado,
    e aponta para a pasta correta do Tesseract embutido.
    """
    if getattr(sys, 'frozen', False):
        # Se estiver rodando como .exe (o PyInstaller extrai os arquivos para a pasta temporária sys._MEIPASS)
        base_path = sys._MEIPASS
    else:
        # Se estiver rodando no VS Code como script normal
        base_path = os.path.dirname(os.path.abspath(__file__))
        
    return os.path.join(base_path, 'Tesseract-OCR', 'tesseract.exe')

# Aplica o caminho dinâmico
pytesseract.pytesseract.tesseract_cmd = obter_caminho_tesseract()

# Qualidade da imagem em DPI (Dots Per Inch). 300 é um bom valor para OCR.
DPI_CONVERSAO = 300

# --- FUNÇÕES MODULARES ---


def corrigir_inclinacao(cinza: np.ndarray) -> np.ndarray:
    """
    Detecta a inclinação do texto e rotaciona a imagem para corrigi-la.
    """
    # Extrai coordenadas de pixels para calcular a inclinação
    _, thresh = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    
    if len(coords) == 0:
        return cinza
        
    angle = cv2.minAreaRect(coords)[-1]

    # Normalização estrita da rotação (evita giros de 90 graus imprevistos)
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90

    (h, w) = cinza.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(cinza, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def pre_processar_imagem_para_ocr(imagem_pil: Image) -> np.ndarray:
    """
    Aplica pré-processamento em uma imagem para melhorar a precisão do OCR.
    Converte para escala de cinza, corrige a inclinação e aplica CLAHE com Otsu.

    Args:
        imagem_pil (Image): A imagem no formato PIL.

    Returns:
        np.ndarray: A imagem processada no formato OpenCV (numpy array).
    """
    imagem_cv = np.array(imagem_pil)
    cinza = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2GRAY)
    
    # Corrige inclinação
    cinza_alinhada = corrigir_inclinacao(cinza)
    
    # Aplica CLAHE para aprimoramento de contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cinza_clahe = clahe.apply(cinza_alinhada)
    
    # Binarização rigorosa Otsu
    _, processada = cv2.threshold(cinza_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return processada


def verificar_assinatura(imagem_processada: np.ndarray) -> bool:
    """
    Localiza a área de assinatura via OCR (múltiplas âncoras) ou usa Fallback Geométrico.
    """
    altura_imagem, largura_imagem = imagem_processada.shape
    limite_y_assinatura = int(altura_imagem * 0.75)

    dados_ocr = pytesseract.image_to_data(imagem_processada, output_type=pytesseract.Output.DICT, lang='por')
    n_boxes = len(dados_ocr['text'])
    
    # 1. MÁSCARA DE EXCLUSÃO DE RODAPÉ (LIMPEZA SELETIVA)
    limite_max_y = altura_imagem
    
    # Âncoras universais (nunca usar nomes próprios de funcionários)
    termos_rodape_exatos = {'pág', 'pag', 'pág.', 'pag.', 'por:'}
    metade_y = int(altura_imagem * 0.5)
    
    for i in range(n_boxes):
        texto_box = dados_ocr['text'][i].strip().lower()
        if not texto_box:
            continue
        
        # Identifica o rodapé de forma exata ou pela palavra-chave 'impresso' ou 'emitido'
        if texto_box in termos_rodape_exatos or 'impresso' in texto_box or 'pág' in texto_box or 'emitido' in texto_box:
            y = dados_ocr['top'][i]
            # Considera apenas se estiver na metade inferior da página (para não apagar o topo por engano)
            if y > metade_y:
                # Puxa a margem 50 pixels para cima (Y - 50) para garantir que engloba a linha toda
                limite_max_y = min(limite_max_y, max(0, y - 50))
                
    # Pinta de branco um retângulo que cubra toda a largura da página, da altura do texto do rodapé encontrado até o fim da imagem.
    if limite_max_y < altura_imagem:
        imagem_processada[limite_max_y:, :] = 255

    # Múltiplas âncoras para não depender de ler apenas a palavra "assinatura"
    ancoras = ['assinatura', 'assina', 'tutor', 'proprietário', 'proprietario', 'responsável', 'responsavel']
    ancora_encontrada = False
    
    for i in range(n_boxes):
        y = dados_ocr['top'][i]
        
        if y < limite_y_assinatura or y >= limite_max_y:
            continue

        texto_box = dados_ocr['text'][i].strip().lower()
        
        if any(ancora in texto_box for ancora in ancoras):
            ancora_encontrada = True
            x, y = dados_ocr['left'][i], dados_ocr['top'][i]
            w, h = dados_ocr['width'][i], dados_ocr['height'][i]
            
            
            roi_h = 150 
            roi_y = max(0, y - roi_h - 20) 
            roi_w = max(w * 3, 200) 
            roi_x = max(0, x - int((roi_w - w) / 2)) 
            
            
            area_assinatura = imagem_processada[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            total_pixels = area_assinatura.size
            if total_pixels == 0:
                continue

            # Limpeza de ruído na ROI
            area_assinatura = cv2.medianBlur(area_assinatura, 3)

            # Inversão para operações morfológicas (fundo preto, traços brancos)
            area_inv = cv2.bitwise_not(area_assinatura)
            
            # 1. Remoção de Linhas Horizontais (Filtro Morfológico)
            kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
            linhas_detectadas = cv2.morphologyEx(area_inv, cv2.MORPH_OPEN, kernel_horizontal)
            area_limpa_inv = cv2.subtract(area_inv, linhas_detectadas)

            # 2. Detecção de Formas (Contornos Estritos)
            contornos, _ = cv2.findContours(area_limpa_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            maior_contorno = 0
            
            for c in contornos:
                area_c = cv2.contourArea(c)
                if area_c > maior_contorno:
                    maior_contorno = area_c
                    
            
            if maior_contorno > 300:
                return True
                
    if not ancora_encontrada:

        # --- FALLBACK GEOMÉTRICO ---
        # Usa o limite_max_y (onde começa o rodapé) para definir o fundo da busca
        corte_y_fallback_fim = limite_max_y
        corte_y_fallback_inicio = max(0, limite_max_y - 250) # Pega 250 pixels acima do rodapé
        corte_x_fallback_inicio = int(largura_imagem * 0.10) # Começa nos 10% da esquerda
        corte_x_fallback_fim = int(largura_imagem * 0.95)
        
        
        if corte_y_fallback_inicio < corte_y_fallback_fim:
            area_fallback = imagem_processada[corte_y_fallback_inicio:corte_y_fallback_fim, corte_x_fallback_inicio:corte_x_fallback_fim]
            total_pixels_fall = area_fallback.size
            
            if total_pixels_fall > 0:
                # Limpeza de ruído na ROI de fallback
                area_fallback = cv2.medianBlur(area_fallback, 3)

                # Inversão para operações morfológicas
                area_fall_inv = cv2.bitwise_not(area_fallback)
                
                # 1. Remoção de Linhas Horizontais
                kernel_horizontal_fall = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
                linhas_detectadas_fall = cv2.morphologyEx(area_fall_inv, cv2.MORPH_OPEN, kernel_horizontal_fall)
                area_limpa_fall_inv = cv2.subtract(area_fall_inv, linhas_detectadas_fall)
                
                # 2. Validação Estrita por Contornos
                contornos_fall, _ = cv2.findContours(area_limpa_fall_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                maior_contorno_fall = 0
                
                for c in contornos_fall:
                    area_c_fall = cv2.contourArea(c)
                    if area_c_fall > maior_contorno_fall:
                        maior_contorno_fall = area_c_fall
                        
                
                if maior_contorno_fall > 300:
                    return True
    
    return False


def limpar_valor(texto: str) -> str:
    """
    Remove rótulos fantasmas capturados pelo OCR e limpa caracteres especiais.
    """
    if not texto:
        return "Não identificado"
    
    # Lista atualizada com TODOS os fantasmas das 416 páginas do Castra-DF
    termos_remover = [
        r'nome completo', r'\(legível\)', r'assinatura', 
        r'responsável pelo animal', r'responsável', r'espécie', 
        r'especie', r'animal', r'cpf', r'paciente', r'data', r'rg',
        r'tutor', r'proprietário', r'nome do cliente', r'raça', r'idade',
        r'cor', r'sexo', r'peso', r'pelagem', r'tipo', r'nome', r'dono',
        r'observações de interesse', r'relatado\s*verbalmente', r'redatado', r'verbanrta',
        r'aproprietarioaresponsavel', r'identificação do animal', r'\bpelo\b', r'\bpe[lfj]o\b', 
        r'impresso.*', r'p[áa]g\w*', r'crmv.*', r'dados do animal', r'doc\.*', 
        r'identifica[çc][ãa]o', r'inscri[çc][ãa]o', r'm[eé]dic[oa].*veterin[aá]ri[oa]', 
        r'animai', r'aniniai', r'peito', r'abaixo\s*identificado', r'abaixo\s*[wvw]entificado',
        r'abaixo\s*aser', r'\bpala\b', r'\bpero\b', r'\bcpe\b'
    ]
    padrao = r'(?i)\b(?:' + '|'.join(termos_remover) + r')\b'
    texto_limpo = re.sub(padrao, '', str(texto))
    
    texto_limpo = re.sub(r'[^A-Za-zÀ-ú ]', '', texto_limpo)
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()
    
    texto_limpo = re.sub(r'\b[A-Za-zÀ-ú]\b', '', texto_limpo)
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()
    
    if not texto_limpo or len(texto_limpo) < 3:
        return "Não identificado"
        
    return texto_limpo


def extrair_apos_delimitador(linha: str) -> str:
    """Separa o rótulo do valor no primeiro delimitador encontrado e retorna o valor."""
    if ':' in linha:
        linha = linha.split(':', 1)[-1]
    elif '-' in linha:
        linha = linha.split('-', 1)[-1]
    return linha.strip()


def validar_cpf(cpf_str: str) -> bool:
    """
    Realiza a validação matemática de um CPF por dígito verificador.
    """
    # 1. Remove qualquer caractere não numérico
    cpf = re.sub(r'\D', '', str(cpf_str))

    # 2. Verifica se o CPF tem 11 dígitos e não são todos iguais
    if len(cpf) != 11 or cpf == cpf[0] * 11:
        return False

    try:
        # 3. Cálculo do primeiro dígito verificador
        soma = 0
        for i in range(9):
            soma += int(cpf[i]) * (10 - i)
        resto = soma % 11
        digito1 = 0 if resto < 2 else 11 - resto

        if digito1 != int(cpf[9]):
            return False

        # 4. Cálculo do segundo dígito verificador
        soma = 0
        for i in range(10):
            soma += int(cpf[i]) * (11 - i)
        resto = soma % 11
        digito2 = 0 if resto < 2 else 11 - resto

        return digito2 == int(cpf[10])

    except (ValueError, IndexError):
        return False


def parsear_dados_do_texto(texto: str) -> dict:
    dados = {
        "Tipo de Termo": "Não identificado",
        "Nome Completo": "Não identificado",
        "CPF": "CPF não identificado",
        "Nome do Animal": "Não identificado",
        "Espécie": "Não identificado"
    }

    linhas = [l.strip() for l in texto.split('\n') if l.strip()]
    
    # --- IDENTIFICAÇÃO DO TIPO DE TERMO (CABEÇALHO) ---
    trecho_topo = " ".join(linhas[:15])
    if re.search(r'(?i)cir[uú]rgico', trecho_topo):
        dados["Tipo de Termo"] = "Cirúrgico"
    elif re.search(r'(?i)antiparasit[aá]rio', trecho_topo):
        dados["Tipo de Termo"] = "Antiparasitário"
    elif re.search(r'(?i)anest[eé]sicos?', trecho_topo):
        dados["Tipo de Termo"] = "Anestésico"

    # 1. BUSCA DO CPF (Âncora Principal - Funciona para todos os layouts)
    cpf_linha_idx = -1
    for i, linha in enumerate(linhas):
        if re.search(r'(?i)\b(?:cep|quadra|lote|conjunto|casa|setor|rua|avenida|residente|bairro|endere[çc]o|tel|telefone|celular|cel)\b', linha):
            if not re.search(r'(?i)\bcpf\b', linha):
                continue
                
        num_limpo = re.sub(r'\D', '', linha)
        if len(num_limpo) >= 11:
            encontrou_valido = False
            for j in range(len(num_limpo) - 10):
                num_cpf = num_limpo[j:j+11]
                if validar_cpf(num_cpf):
                    dados["CPF"] = f"{num_cpf[:3]}.{num_cpf[3:6]}.{num_cpf[6:9]}-{num_cpf[9:]}"
                    encontrou_valido = True
                    cpf_linha_idx = i
                    print(f"    [LOG] CPF matemático válido capturado: {dados['CPF']}")
                    break
            
            if encontrou_valido:
                break
            elif dados["CPF"] == "CPF não identificado" and len(num_limpo) == 11:
                dados["CPF"] = "CPF Inválido"
                cpf_linha_idx = i

    # --- 2. VIA RÁPIDA: NOVO LAYOUT ESTRUTURADO (OMNI / Lista) ---
    for linha in linhas:
        match_animal = re.search(r'(?i)^Animal:\s*(?:\d+\s*-\s*)?(.+)', linha)
        if match_animal and dados["Nome do Animal"] == "Não identificado":
            dados["Nome do Animal"] = limpar_valor(match_animal.group(1))

        match_especie = re.search(r'(?i)^Esp[eéè]cie:\s*(.+)', linha)
        if match_especie and dados["Espécie"] == "Não identificado":
            dados["Espécie"] = limpar_valor(match_especie.group(1))

        match_tutor = re.search(r'(?i)^Respons[aá]vel:\s*(?:\d+\s*-\s*)?(.+)', linha)
        if match_tutor and dados["Nome Completo"] == "Não identificado":
            dados["Nome Completo"] = limpar_valor(match_tutor.group(1))
            
        match_nome = re.search(r'(?i)^Nome:\s*(.+)', linha)
        if match_nome and dados["Nome Completo"] == "Não identificado":
            candidato = limpar_valor(match_nome.group(1))
            if candidato != "Não identificado" and len(candidato.split()) > 1:
                dados["Nome Completo"] = candidato

        match_animal_inline = re.search(r'(?i)\bNome\s+([^,]+),\s*esp[eéè]cie', linha)
        if match_animal_inline and dados["Nome do Animal"] == "Não identificado":
            dados["Nome do Animal"] = limpar_valor(match_animal_inline.group(1))

    # --- 3. FALLBACK: LAYOUT ANTIGO (Texto Corrido / Parágrafos) ---
    if dados["Nome Completo"] == "Não identificado":
        for i, linha in enumerate(linhas):
            if re.search(r'(?i)(?:observaç[õo]es|impresso|p[áa]g|assinatura|pelo\s+animal|crmv)', linha):
                continue
            if re.search(r'(?i)\b(?:felin[oa]|canin[oa]|f[êe]mea|macho|nascid[oa]|pelagem|srd|kg|chip)\b', linha):
                continue
                
            match_tutor = re.search(r'(?i)^(?:nome.*completo|nome\s+do\s+(?:tutor|respons[áa]vel|propriet[áa]rio)|tutor|propriet[áa]rio|respons[áa]vel|cliente|nome\b)[\s:\-.]*(.*)', linha)
            if match_tutor:
                candidato = match_tutor.group(1).strip()
                nome_limpo = limpar_valor(candidato)
                if nome_limpo != "Não identificado":
                    dados["Nome Completo"] = nome_limpo
                    break
                elif i + 1 < len(linhas):
                    nome_limpo_abaixo = limpar_valor(linhas[i+1])
                    if nome_limpo_abaixo != "Não identificado":
                        dados["Nome Completo"] = nome_limpo_abaixo
                        break

        if dados["Nome Completo"] == "Não identificado" and cpf_linha_idx > 0 and dados["CPF"] not in ["CPF Inválido", "CPF não identificado"]:
            for j in range(cpf_linha_idx - 1, max(-1, cpf_linha_idx - 4), -1):
                candidato = limpar_valor(linhas[j])
                if candidato != "Não identificado" and not re.search(r'(?i)(?:cpf|rg|doc)', linhas[j]):
                    dados["Nome Completo"] = candidato
                    break

    if dados["Nome do Animal"] == "Não identificado" or dados["Espécie"] == "Não identificado":
        for i, linha in enumerate(linhas):
            if re.search(r'(?i)(?:identifica[çc][ãa]o|dados\s+do\s+animal|paciente|esp[eéè]cie)', linha):
                busca_area = " \n ".join(linhas[i:i+7]) 
                if "identifica" in linha.lower() and not re.search(r'(?i)(?:esp[eéè]cie|ra[çc]a|pelagem|sexo|idade|f[êe]mea|macho)', busca_area):
                    continue
                
                match_animal = re.search(r'(?i)\bnome\b(?!\s+completo)[\s:\-.]*([A-Za-zÀ-ú\s]+?)(?:[\n,\(]|esp[eéè]cie|ra[çc]a|sexo|idade|f[êe]mea|macho|nascid[oa]|pelagem|kg|peso|\bem\b)', busca_area)
                if match_animal and dados["Nome do Animal"] == "Não identificado":
                    dados["Nome do Animal"] = limpar_valor(match_animal.group(1))
                    
                match_especie = re.search(r'(?i)esp[eéè]cie[\s:\-.]*([A-Za-zÀ-ú\s]+?)(?:[\n,]|$|ra[çc]a|sexo|idade|f[êe]mea|macho|nascid[oa]|pelagem)', busca_area)
                if match_especie and dados["Espécie"] == "Não identificado":
                    dados["Espécie"] = limpar_valor(match_especie.group(1))
                
                if dados["Nome do Animal"] != "Não identificado" or dados["Espécie"] != "Não identificado":
                    break
                    
    if dados["Espécie"] == "Não identificado":
        for linha in linhas:
            match_especie_fallback = re.search(r'(?i)esp[eéè]cie\s*[:\-]?\s*([A-Za-zÀ-ú\s]+)', linha)
            if match_especie_fallback:
                especie_limpa = limpar_valor(match_especie_fallback.group(1))
                if especie_limpa != "Não identificado":
                    dados["Espécie"] = especie_limpa
                    break

    # --- LIMPEZA ESPECÍFICA DE RUÍDOS NO NOME DO ANIMAL ---
    if dados["Nome do Animal"] not in ["Não identificado"]:
        # Remove 'Kg', 'Em', 'Canina Fêmea', parênteses e outras sujeiras específicas
        animal_limpo = re.sub(r'(?i)(\bkg\b|\bem\b|\bcanin[oa]\b|\bfelin[oa]\b|\bf[eê]mea\b|\bmach[oa]\b|\bpeso\b|\(|\))', '', dados["Nome do Animal"])
        animal_limpo = re.sub(r'\s+', ' ', animal_limpo).strip()
        dados["Nome do Animal"] = animal_limpo if len(animal_limpo) >= 2 else "Não identificado"

    # --- LIMPEZA ESPECÍFICA NA ESPÉCIE ---
    if dados["Espécie"] not in ["Não identificado"]:
        if re.search(r'(?i)canin[oa]?', dados["Espécie"]):
            dados["Espécie"] = "Canina"
        elif re.search(r'(?i)felin[oa]?', dados["Espécie"]):
            dados["Espécie"] = "Felina"
        else:
            especie_limpa = re.sub(r'(?i)\b(f[eê]mea|macho)\b', '', dados["Espécie"])
            especie_limpa = re.sub(r'\s+', ' ', especie_limpa).strip()
            dados["Espécie"] = especie_limpa if len(especie_limpa) >= 2 else "Não identificado"

    # --- VALIDAÇÃO FINAL ---
    if dados["Nome Completo"] != "Não identificado" and dados["Nome do Animal"] == dados["Nome Completo"]:
        dados["Nome do Animal"] = "Não identificado"

    excecoes_prep = ['de', 'da', 'do', 'das', 'dos', 'e']
    for chave in ["Nome Completo", "Nome do Animal", "Espécie"]:
        if dados[chave] not in ["Não identificado", "CPF não identificado", "CPF Inválido"]:
            palavras = str(dados[chave]).split()
            palavras_formatadas = [
                p.capitalize() if p.lower() not in excecoes_prep else p.lower()
                for p in palavras
            ]
            dados[chave] = " ".join(palavras_formatadas)

    return dados


def limpar_dados_planilha(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpezas padronizadas (Espécie, Capitalização) em um DataFrame."""
    if df.empty:
        return df

    df_limpo = df.copy()

    # 1. Limpeza da Coluna 'Espécie'
    if 'Espécie' in df_limpo.columns:
        def limpar_especie_valor(especie):
            if pd.isna(especie) or not isinstance(especie, str) or especie.strip().lower() in ["", "---", "não identificado"]:
                return "Não identificado"
            
            especie_str = str(especie)
            if re.search(r'(?i)canin[oa]?', especie_str):
                return "Canina"
            elif re.search(r'(?i)felin[oa]?', especie_str):
                return "Felina"
            else:
                especie_limpa = re.sub(r'(?i)\b(f[eê]mea|macho)\b', '', especie_str)
                especie_limpa = re.sub(r'\s+', ' ', especie_limpa).strip()
                return especie_limpa if len(especie_limpa) >= 2 else "Não identificado"

        df_limpo['Espécie'] = df_limpo['Espécie'].apply(limpar_especie_valor)

    # 2. Capitalização Inteligente
    excecoes_prep = ['de', 'da', 'do', 'das', 'dos', 'e']
    for chave in ["Nome Completo", "Nome do Animal", "Espécie"]:
        if chave in df_limpo.columns:
            def capitalizar_inteligentemente(texto):
                if not isinstance(texto, str) or texto.strip().lower() in ["", "---", "não identificado", "cpf não identificado", "cpf inválido"]:
                    return texto
                palavras = texto.split()
                palavras_formatadas = [p.capitalize() if p.lower() not in excecoes_prep else p.lower() for p in palavras]
                return " ".join(palavras_formatadas)
            
            df_limpo[chave] = df_limpo[chave].apply(capitalizar_inteligentemente)

    return df_limpo


class RedirecionadorTerminal:
    """Classe que captura os 'prints' do sistema e os envia para a interface gráfica."""
    def __init__(self, widget_texto):
        self.widget_texto = widget_texto

    def write(self, string):
        # Utiliza o .after() para ser thread-safe (não travar o processamento do OCR)
        self.widget_texto.after(0, self._inserir, string)

    def _inserir(self, string):
        self.widget_texto.insert("end", string)
        self.widget_texto.see("end") # Faz o scroll automático ir para baixo

    def flush(self):
        pass


def processar_pdf_formularios(caminho_pdf: str, progresso_callback=None, itens_revisao_callback=None):
    """
    Processa o PDF, extrai os dados, aplica OCR de 2ª chance e alimenta os Bancos de Dados.
    """
    nome_arquivo = os.path.basename(caminho_pdf)
    diretorio_base = os.path.dirname(caminho_pdf)
    revisao_manual_path = os.path.join(diretorio_base, 'REVISAO_MANUAL_CASTRA.xlsx')
    
    mapa_arquivos = {
        'Cirúrgico': 'BANCO_MESTRE_CIRURGICO.xlsx',
        'Antiparasitário': 'BANCO_MESTRE_ANTIPARASITARIO.xlsx',
        'Anestésico': 'BANCO_MESTRE_ANESTESICO.xlsx',
        'Não identificado': 'BANCO_MESTRE_NAO_IDENTIFICADO.xlsx'
    }

    print(f"📂 ARQUIVO: {nome_arquivo}")

    try:
        pdf = pdfium.PdfDocument(caminho_pdf)
        total_paginas = len(pdf)
        print(f"📑 Total de páginas identificadas: {total_paginas}\n")
    except Exception as e:
        print(f"❌ Erro ao ler o arquivo: {e}")
        return

    # --- LÓGICA DE RESUMO AUTOMÁTICO ---
    paginas_processadas = set()
    
    # Verifica páginas já concluídas em todos os Bancos Mestres
    for arquivo_banco in mapa_arquivos.values():
        banco_mestre_path = os.path.join(diretorio_base, arquivo_banco)
        if os.path.exists(banco_mestre_path):
            try:
                df_m = pd.read_excel(banco_mestre_path)
                if 'Arquivo Origem' in df_m.columns and 'Página' in df_m.columns:
                    paginas_processadas.update(df_m[df_m['Arquivo Origem'] == nome_arquivo]['Página'].dropna().astype(int).tolist())
            except Exception: pass
        
    # Verifica páginas já concluídas no Relatório de Erros
    if os.path.exists(revisao_manual_path):
        try:
            df_r = pd.read_excel(revisao_manual_path)
            if 'Arquivo Origem' in df_r.columns and 'Página' in df_r.columns:
                paginas_processadas.update(df_r[df_r['Arquivo Origem'] == nome_arquivo]['Página'].dropna().astype(int).tolist())
        except Exception: pass

    if len(paginas_processadas) >= total_paginas:
        print(f"✅ Arquivo já foi 100% processado anteriormente. Pulando...\n")
        if progresso_callback:
            progresso_callback(total_paginas, total_paginas)
        return
    elif paginas_processadas:
        print(f"🔄 Retomando processamento... {len(paginas_processadas)} páginas já lidas foram ignoradas.\n")

    dados_coletados = []
    escala = DPI_CONVERSAO / 72.0

    def salvar_dados_parciais():
        """Função aninhada para salvar dados no disco periodicamente e liberar RAM"""
        if not dados_coletados:
            return
            
        print("💾 Sincronizando dados com as planilhas do sistema...")
        df_novo = pd.DataFrame(dados_coletados)
        df_mestre_check_all = pd.DataFrame()
        
        # 1. Atualizar Bancos Mestres (Separados por Tipo)
        tipos_presentes = df_novo['Tipo de Termo'].unique()
        
        for tipo in tipos_presentes:
            df_tipo = df_novo[df_novo['Tipo de Termo'] == tipo].copy()
            nome_arquivo_banco = mapa_arquivos.get(tipo, 'BANCO_MESTRE_NAO_IDENTIFICADO.xlsx')
            banco_path = os.path.join(diretorio_base, nome_arquivo_banco)
            
            df_mestre_check = pd.DataFrame()
            if os.path.exists(banco_path):
                try:
                    df_mestre = pd.read_excel(banco_path)
                    df_mestre_check = df_mestre.copy()
                    df_final = pd.concat([df_mestre, df_tipo], ignore_index=True)
                except Exception:
                    df_final = df_tipo.copy()
            else:
                df_final = df_tipo.copy()
                
            df_mestre_check_all = pd.concat([df_mestre_check_all, df_mestre_check], ignore_index=True)

            # Remove Duplicados
            mascara_validos = (df_final['CPF'] != 'CPF não identificado') & (df_final['CPF'] != 'Não identificado') & (df_final['CPF'] != 'CPF Inválido')
            colunas_dup = [c for c in ['Nome Completo', 'CPF', 'Nome do Animal', 'Tipo de Termo'] if c in df_final.columns]
            duplicados = df_final[mascara_validos].duplicated(subset=colunas_dup, keep='last')
            df_final = df_final.drop(duplicados[duplicados].index)
            
            colunas_ordem = ['Arquivo Origem', 'Página', 'Tipo de Termo', 'Nome Completo', 'CPF', 'Nome do Animal', 'Espécie', 'Assinatura Presente']
            df_final = df_final[[col for col in colunas_ordem if col in df_final.columns]]
            df_final = df_final.sort_values(by=['Arquivo Origem', 'Página'], ascending=[True, True])
            
            try:
                df_final.to_excel(banco_path, index=False, engine='openpyxl')
            except Exception as e:
                print(f"❌ Erro ao salvar base acumulada ({tipo}): {e}")

        # 2. Relatório de Inconsistências (Revisão Humana)
        erros_list = []
        vistos_lote = {}
        for _, row in df_novo.iterrows():
            motivos = []
            if row.get('Tipo de Termo') == "Não identificado": motivos.append("Tipo não identificado")
            if row.get('Nome Completo') == "Não identificado": motivos.append("Nome Ausente/Ilegível")
            if row.get('CPF') in ["CPF não identificado", "CPF Inválido", "Não identificado"]: motivos.append("CPF Ausente/Inválido")
            if row.get('Nome do Animal') == "Não identificado": motivos.append("Animal Ausente/Ilegível")
            if row.get('Espécie') == "Não identificado": motivos.append("Espécie Ausente/Ilegível")
            if row.get('Assinatura Presente') == "Não": motivos.append("Assinatura Pendente")
            
            is_dup = False
            cpf_valido = row.get('CPF') not in ["CPF não identificado", "CPF Inválido", "Não identificado"]
            if cpf_valido:
                cpf = row.get('CPF')
                animal = row.get('Nome do Animal')
                nome = row.get('Nome Completo')
                tipo = row.get('Tipo de Termo')
                chave_dup = (nome, cpf, animal, tipo)
                
                if not df_mestre_check_all.empty and all(c in df_mestre_check_all.columns for c in ['Nome Completo', 'CPF', 'Nome do Animal', 'Tipo de Termo']):
                    match = (df_mestre_check_all['Nome Completo'] == nome) & (df_mestre_check_all['CPF'] == cpf) & (df_mestre_check_all['Nome do Animal'] == animal) & (df_mestre_check_all['Tipo de Termo'] == tipo)
                    if match.any(): 
                        is_dup = True
                        row_match = df_mestre_check_all[match].iloc[0]
                        motivos.append(f"Registro Duplicado (Arq: {row_match.get('Arquivo Origem', '?')}, Pág: {row_match.get('Página', '?')})")
                
                if not is_dup:
                    if chave_dup in vistos_lote:
                        arq_dup, pag_dup = vistos_lote[chave_dup]
                        motivos.append(f"Registro Duplicado (Arq: {arq_dup}, Pág: {pag_dup})")
                    else:
                        vistos_lote[chave_dup] = (row.get('Arquivo Origem'), row.get('Página'))

            if motivos:
                dict_erro = row.to_dict()
                dict_erro['O que falta corrigir?'] = " | ".join(motivos)
                erros_list.append(dict_erro)

        if itens_revisao_callback:
            itens_revisao_callback(len(erros_list))

        if erros_list:
            df_erros = pd.DataFrame(erros_list)
            if os.path.exists(revisao_manual_path):
                try:
                    df_revisao_antigo = pd.read_excel(revisao_manual_path)
                    if 'Motivo da Falha' in df_revisao_antigo.columns and 'O que falta corrigir?' not in df_revisao_antigo.columns:
                        df_revisao_antigo = df_revisao_antigo.rename(columns={'Motivo da Falha': 'O que falta corrigir?'})
                    df_revisao_final = pd.concat([df_revisao_antigo, df_erros], ignore_index=True)
                except Exception:
                    df_revisao_final = df_erros
            else:
                df_revisao_final = df_erros
            
            try:
                df_revisao_final = df_revisao_final.drop_duplicates(subset=['CPF', 'Nome do Animal', 'O que falta corrigir?', 'Arquivo Origem'], keep='last')
                df_revisao_final = df_revisao_final.sort_values(by=['Arquivo Origem', 'Página'], ascending=[True, True])
                df_revisao_final.to_excel(revisao_manual_path, index=False, engine='openpyxl')
            except Exception: pass

        # Limpa os dados parciais da memória após salvá-los no disco
        dados_coletados.clear()
    
    for i in range(total_paginas):
        pagina_atual = i + 1
        if progresso_callback:
            progresso_callback(i, total_paginas)

        if pagina_atual in paginas_processadas:
            print(f"⏭️ PÁGINA {pagina_atual} / {total_paginas} (Já processada, ignorando...)")
            continue

        print(f"📄 PÁGINA {pagina_atual} / {total_paginas}")
        print("━" * 45)

        pagina = pdf[i]
        bitmap = pagina.render(scale=escala)
        imagem_pil = bitmap.to_pil()
        pagina.close()

        # 1. Pré-processamento e corte
        imagem_processada = pre_processar_imagem_para_ocr(imagem_pil)
        h_img, w_img = imagem_processada.shape
        corte_y = int(h_img * 0.05)
        imagem_dados = imagem_processada[corte_y:h_img, 0:w_img]

        # 2. OCR (Primeira Passagem)
        texto_extraido = pytesseract.image_to_string(imagem_dados, lang='por')

        # 3. Parseamento
        dados_pagina = parsear_dados_do_texto(texto_extraido)
        
        # 4. INTELIGÊNCIA OCR: Segunda Chance (Inversão de Cores)
        if dados_pagina["Nome Completo"] == "Não identificado":
            print("  🔄 Tratando imagem para melhorar legibilidade...")
            imagem_invertida = cv2.bitwise_not(imagem_dados)
            texto_inv = pytesseract.image_to_string(imagem_invertida, lang='por')
            dados_inv = parsear_dados_do_texto(texto_inv)
            
            if dados_inv["Nome Completo"] != "Não identificado":
                dados_pagina["Nome Completo"] = dados_inv["Nome Completo"]

        dados_pagina["Página"] = pagina_atual
        dados_pagina["Arquivo Origem"] = nome_arquivo

        # 5. Assinatura
        assinatura_presente = verificar_assinatura(imagem_processada)
        dados_pagina["Assinatura Presente"] = "Sim" if assinatura_presente else "Não"

        status_ass = "✅ Identificada" if assinatura_presente else "❌ Não identificada"
        print(f"  📄 Tipo:       {dados_pagina.get('Tipo de Termo', '')}")
        print(f"  👤 Tutor:      {dados_pagina.get('Nome Completo', '')}")
        print(f"  🪪 CPF:        {dados_pagina.get('CPF', '')}")
        print(f"  🐾 Pet:        {dados_pagina.get('Nome do Animal', '')} ({dados_pagina.get('Espécie', '')})")
        print(f"  ✍️ Assinatura: {status_ass}")
        print("━" * 45 + "\n")
        dados_coletados.append(dados_pagina)

        # Limpa da memória os objetos pesados do processamento atual
        del imagem_pil
        del imagem_processada
        del bitmap

        # Salva o progresso a cada 10 páginas para garantir o resume em caso de interrupção
        if pagina_atual % 10 == 0:
            salvar_dados_parciais()

    if progresso_callback:
        progresso_callback(total_paginas, total_paginas)

    # Salva eventuais dados restantes no fim do arquivo
    salvar_dados_parciais()


def iniciar_gui():
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    root = ctk.CTk()
    root.title("Castra-DF - Processador de Formulários Mestre")
    root.geometry("1100x750")
    
    style = ttk.Style(root)
    style.theme_use('clam')
    
    estado_gui = {
        "df_banco_atual": pd.DataFrame(),
        "caminho_banco_atual": "",
        "alterado": False,
        "df_erros_atual": pd.DataFrame(),
        "caminho_erros_atual": "",
        "alterado_erros": False,
        "timer_autosave": None
    }

    def remover_duplicidades_df(df):
        if df.empty:
            return df, 0
        mascara_validos = (df['CPF'] != 'CPF não identificado') & (df['CPF'] != 'Não identificado') & (df['CPF'] != 'CPF Inválido')
        cols_subset = [c for c in ['Nome Completo', 'CPF', 'Nome do Animal', 'Tipo de Termo'] if c in df.columns]
        if not cols_subset:
            return df, 0
        duplicados = df[mascara_validos].duplicated(subset=cols_subset, keep='first')
        qtd = duplicados.sum()
        df_limpo = df.drop(duplicados[duplicados].index).reset_index(drop=True)
        return df_limpo, qtd

    def salvar_banco_atual():
        if estado_gui.get("alterado") and estado_gui.get("caminho_banco_atual"):
            try:
                df_to_save = estado_gui["df_banco_atual"]
                df_to_save = df_to_save.sort_values(by=['Arquivo Origem', 'Página'], ascending=[True, True])
                df_to_save.to_excel(estado_gui["caminho_banco_atual"], index=False, engine='openpyxl')
                estado_gui["alterado"] = False
                print(f"Banco de dados salvo com sucesso: {estado_gui['caminho_banco_atual']}")
            except Exception as e:
                print(f"Erro ao salvar banco de dados: {e}")
                messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar as alterações:\n{e}")

    def salvar_erros_atual():
        if estado_gui.get("alterado_erros") and estado_gui.get("caminho_erros_atual"):
            try:
                df_to_save = estado_gui["df_erros_atual"]
                df_to_save = df_to_save.sort_values(by=['Arquivo Origem', 'Página'], ascending=[True, True])
                df_to_save.to_excel(estado_gui["caminho_erros_atual"], index=False, engine='openpyxl')
                estado_gui["alterado_erros"] = False
                print(f"Relatório de erros salvo com sucesso: {estado_gui['caminho_erros_atual']}")
            except Exception as e:
                print(f"Erro ao salvar erros: {e}")
                messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar os erros:\n{e}")

    def autosave():
        if estado_gui.get("alterado"):
            salvar_banco_atual()
        if estado_gui.get("alterado_erros"):
            salvar_erros_atual()
        estado_gui["timer_autosave"] = root.after(60000, autosave) # Autosave a cada 60s

    frame_top = ctk.CTkFrame(root, fg_color="transparent")
    frame_top.pack(fill=tk.X, padx=15, pady=(15, 0))
    
    def change_appearance_mode_event(new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)
        bg_color = "#2b2b2b" if new_appearance_mode == "Dark" else "#ebebeb"
        text_color = "white" if new_appearance_mode == "Dark" else "black"
        head_bg = "#1f538d"
        
        style.configure("Treeview", background=bg_color, foreground=text_color, fieldbackground=bg_color, borderwidth=0, rowheight=25)
        style.map('Treeview', background=[('selected', '#1f538d')], foreground=[('selected', 'white')])
        style.configure("Treeview.Heading", background=head_bg, foreground="white", font=("Segoe UI", 10, "bold"))

    modo_label = ctk.CTkLabel(frame_top, text="Tema:", font=("Segoe UI", 12))
    modo_label.pack(side=tk.RIGHT, padx=(10, 5))
    
    modo_menu = ctk.CTkOptionMenu(frame_top, values=["Dark", "Light", "System"], command=change_appearance_mode_event)
    modo_menu.pack(side=tk.RIGHT)
    modo_menu.set("Dark")
    change_appearance_mode_event("Dark")
    
    def acao_sincronizar():
        atualizar_tabelas()
        messagebox.showinfo("Sincronização", "As planilhas foram recarregadas e a interface foi sincronizada com sucesso!")

    btn_sync_global = ctk.CTkButton(frame_top, text="🔄 Recarregar / Sincronizar", command=lambda: acao_sincronizar(), font=("Segoe UI", 12, "bold"), fg_color="#5bc0de", hover_color="#31b0d5", text_color="black")
    btn_sync_global.pack(side=tk.LEFT, padx=10)
    
    notebook = ctk.CTkTabview(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
    
    tab_processar = notebook.add('⚙️ Processar PDFs')
    tab_banco = notebook.add('🔍 Consultar Cadastros Concluídos')
    tab_erros = notebook.add('📋 Pendências de Revisão')
    
    # --- ABA 1: PROCESSAMENTO ---
    ctk.CTkLabel(tab_processar, text="Selecione o arquivo PDF ou Pasta para processamento em lote:", font=("Segoe UI", 14, "bold")).pack(anchor=tk.W, padx=20, pady=(10, 10))
    
    frame_arquivo = ctk.CTkFrame(tab_processar)
    frame_arquivo.pack(fill=tk.X, padx=20, pady=5)
    
    entrada_pdf = ctk.CTkEntry(frame_arquivo, font=("Segoe UI", 12))
    entrada_pdf.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10), pady=10)
    
    def selecionar_pdf():
        caminho = filedialog.askopenfilename(title="Selecione o PDF", filetypes=[("Arquivos PDF", "*.pdf")])
        if caminho:
            entrada_pdf.delete(0, "end")
            entrada_pdf.insert(0, caminho)
            atualizar_tabelas() # Tenta carregar os bancos do diretório do PDF selecionado
            
    def selecionar_pasta():
        caminho = filedialog.askdirectory(title="Selecione a Pasta com PDFs")
        if caminho:
            entrada_pdf.delete(0, "end")
            entrada_pdf.insert(0, caminho)
            atualizar_tabelas()
            
    ctk.CTkButton(frame_arquivo, text="Procurar Pasta", command=selecionar_pasta).pack(side=tk.RIGHT, padx=(5, 10), pady=10)
    ctk.CTkButton(frame_arquivo, text="Procurar Arquivo", command=selecionar_pdf).pack(side=tk.RIGHT, pady=10)
    
    # Dashboard
    frame_dash = ctk.CTkFrame(tab_processar)
    frame_dash.pack(fill=tk.X, padx=20, pady=15)
    
    barra_progresso = ctk.CTkProgressBar(frame_dash)
    barra_progresso.pack(fill=tk.X, padx=15, pady=(15, 10))
    barra_progresso.set(0)
    
    lbl_progresso = ctk.CTkLabel(frame_dash, text="Pronto para iniciar.", font=("Segoe UI", 12, "italic"), text_color="gray")
    lbl_progresso.pack(anchor=tk.W, padx=15, pady=(0, 10))
    
    lbl_revisao = ctk.CTkLabel(frame_dash, text="Itens que precisam de revisão humana: 0", font=("Segoe UI", 14, "bold"), text_color="#f38ba8")
    lbl_revisao.pack(anchor=tk.E, side=tk.BOTTOM, padx=15, pady=(0, 15))
    
    btn_processar = ctk.CTkButton(tab_processar, text="🚀 LER FORMULÁRIOS AGORA", font=("Segoe UI", 14, "bold"), height=40, command=lambda: iniciar_thread())
    btn_processar.pack(fill=tk.X, padx=20, pady=10)
    
    # Console
    ctk.CTkLabel(tab_processar, text="Console de Execução:", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, padx=20, pady=(10, 5))
    texto_console = ctk.CTkTextbox(tab_processar, font=("Consolas", 12), wrap="word")
    texto_console.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
    
    sys.stdout = RedirecionadorTerminal(texto_console)
    
    total_revisao = [0]

    def cb_revisao(qnt):
        total_revisao[0] += qnt
        lbl_revisao.configure(text=f"Itens que precisam de revisão humana (Total Acumulado): {total_revisao[0]}")
        root.update_idletasks()
        
    def iniciar_thread():
        caminho_input = entrada_pdf.get()
        if not caminho_input or not os.path.exists(caminho_input):
            messagebox.showerror("Erro", "Por favor, selecione um arquivo ou pasta válida.")
            return
            
        btn_processar.configure(state="disabled")
        barra_progresso.set(0)
        total_revisao[0] = 0
        lbl_revisao.configure(text="Itens que precisam de revisão humana: 0")
        
        def run():
            try:
                if os.path.isdir(caminho_input):
                    arquivos_pdf = [os.path.join(caminho_input, f) for f in os.listdir(caminho_input) if f.lower().endswith('.pdf')]
                    if not arquivos_pdf:
                        root.after(0, lambda: messagebox.showwarning("Aviso", "Nenhum PDF encontrado na pasta selecionada."))
                        return
                else:
                    arquivos_pdf = [caminho_input]

                total_pdfs = len(arquivos_pdf)
                for idx, pdf in enumerate(arquivos_pdf):
                    
                    def cb_progresso(atual, total, idx=idx, total_pdfs=total_pdfs):
                        pct = atual / total if total > 0 else 0
                        barra_progresso.set(pct)
                        lbl_progresso.configure(text=f"[Arquivo {idx+1}/{total_pdfs}] Processando página {atual} de {total} ({pct*100:.1f}%)")
                        root.update_idletasks()
                        
                    processar_pdf_formularios(pdf, progresso_callback=cb_progresso, itens_revisao_callback=cb_revisao)
                    
                root.after(0, lambda: messagebox.showinfo("Concluído", "Tudo pronto! O sistema leu todas as páginas.\n\nAgora, confira se há pendências na aba ao lado antes de fechar."))
                root.after(0, atualizar_tabelas)
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("Erro de Execução", f"Ocorreu um erro inesperado:\n{e}"))
            finally:
                root.after(0, lambda: btn_processar.configure(state="normal"))
                
        threading.Thread(target=run, daemon=True).start()
        
    # --- ABAS DE TABELAS (Bancos e Erros) ---
    def configurar_treeview(parent):
        frame_tv = ctk.CTkFrame(parent)
        frame_tv.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scroll_y = ttk.Scrollbar(frame_tv)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x = ttk.Scrollbar(frame_tv, orient=tk.HORIZONTAL)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        tv = ttk.Treeview(frame_tv, yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        tv.pack(fill=tk.BOTH, expand=True)
        
        scroll_y.config(command=tv.yview)
        scroll_x.config(command=tv.xview)
        
        return tv

    # --- BARRA DE PESQUISA (Aba Banco de Dados) ---
    frame_pesquisa = ctk.CTkFrame(tab_banco)
    frame_pesquisa.pack(fill=tk.X, padx=10, pady=(10, 0))
    
    ctk.CTkLabel(frame_pesquisa, text="Banco:", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=(10, 5), pady=10)
    tipo_banco_var = ctk.StringVar(value="Cirúrgico")
    
    def ao_mudar_banco(_):
        if estado_gui.get("alterado"):
            salvar_banco_atual()
        atualizar_tabelas()
        
    combo_banco = ctk.CTkComboBox(frame_pesquisa, values=["Cirúrgico", "Antiparasitário", "Anestésico", "Não identificado"], variable=tipo_banco_var, command=ao_mudar_banco, width=150)
    combo_banco.pack(side=tk.LEFT, padx=5, pady=10)
    
    lbl_total_banco = ctk.CTkLabel(frame_pesquisa, text="Registros: 0", font=("Segoe UI", 12, "bold"), text_color="#5bc0de")
    lbl_total_banco.pack(side=tk.LEFT, padx=(5, 15), pady=10)

    ctk.CTkLabel(frame_pesquisa, text="Pesquisar por Nome/CPF:", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=(20, 5), pady=10)
    entrada_pesquisa = ctk.CTkEntry(frame_pesquisa, font=("Segoe UI", 12), width=250)
    entrada_pesquisa.pack(side=tk.LEFT, padx=5, pady=10)

    def pesquisar_banco(event=None):
        termo = entrada_pesquisa.get().strip().lower()
        if not termo:
            limpar_filtro()
            return

        if estado_gui["df_banco_atual"].empty:
            return

        try:
            df = estado_gui["df_banco_atual"]
            mask = df['Nome Completo'].astype(str).str.lower().str.contains(termo) | \
                   df['CPF'].astype(str).str.lower().str.contains(termo)
            df_filtrado = df[mask]

            for row in tv_banco.get_children(): tv_banco.delete(row)
            for idx, row in df_filtrado.iterrows():
                tags = ()
                cpf_val = str(row.get('CPF', ''))
                if cpf_val not in ["---", "Não identificado", "CPF não identificado", "CPF Inválido", ""] and not validar_cpf(cpf_val):
                    tags = ('cpf_invalido_matematico',)
                tv_banco.insert("", "end", iid=str(idx), values=list(row), tags=tags)
            atualizar_contadores_tabelas()
        except Exception as e:
            print(f"Erro ao pesquisar: {e}")

    entrada_pesquisa.bind("<KeyRelease>", pesquisar_banco)

    def limpar_filtro():
        entrada_pesquisa.delete(0, "end")
        atualizar_tabelas()

    def btn_limpar_duplicidades_click():
        if estado_gui["df_banco_atual"].empty:
            return
        df_limpo, qtd = remover_duplicidades_df(estado_gui["df_banco_atual"])
        if qtd > 0:
            if messagebox.askyesno("Limpar Duplicidades", f"Foram encontrados {qtd} registros duplicados nesta base.\nDeseja removê-los?"):
                estado_gui["df_banco_atual"] = df_limpo
                estado_gui["alterado"] = True
                salvar_banco_atual()
                atualizar_tabelas()
                messagebox.showinfo("Sucesso", f"{qtd} duplicidades removidas.")
        else:
            messagebox.showinfo("Verificação", "Nenhuma duplicidade encontrada na base atual.")

    def excluir_linha_banco():
        selecionados = tv_banco.selection()
        if not selecionados:
            messagebox.showwarning("Aviso", "Selecione uma ou mais linhas na tabela para excluir.")
            return
        if messagebox.askyesno("Confirmar Exclusão", "Deseja realmente excluir permanentemente a(s) linha(s) selecionada(s) do Banco de Dados?"):
            indices = [int(item) for item in selecionados]
            estado_gui["df_banco_atual"] = estado_gui["df_banco_atual"].drop(indices).reset_index(drop=True)
            estado_gui["alterado"] = True
            salvar_banco_atual()
            atualizar_tabelas()

    def carregar_planilha_avulsa():
        if estado_gui.get("alterado"):
            salvar_banco_atual()
        if estado_gui.get("alterado_erros"):
            salvar_erros_atual()
            
        pasta = filedialog.askdirectory(title="Selecione a Pasta com as Planilhas")
        if pasta:
            # Altera o diretório de trabalho global do aplicativo
            entrada_pdf.delete(0, "end")
            entrada_pdf.insert(0, pasta)
            # Deixa o sistema organizar e colocar cada arquivo em sua respectiva aba automaticamente
            atualizar_tabelas()
            messagebox.showinfo("Planilhas Carregadas", "As planilhas do diretório selecionado foram carregadas e distribuídas corretamente nas abas.")

    ctk.CTkButton(frame_pesquisa, text="Limpar Filtro", command=limpar_filtro, width=100).pack(side=tk.LEFT, padx=5, pady=10)
    ctk.CTkButton(frame_pesquisa, text="Limpar Duplicidades", command=btn_limpar_duplicidades_click, width=150, fg_color="#d9534f", hover_color="#c9302c").pack(side=tk.LEFT, padx=5, pady=10)
    ctk.CTkButton(frame_pesquisa, text="Excluir Linha", command=excluir_linha_banco, width=120, fg_color="#d9534f", hover_color="#c9302c").pack(side=tk.LEFT, padx=5, pady=10)
    ctk.CTkButton(frame_pesquisa, text="Carregar Planilhas", command=carregar_planilha_avulsa, width=120).pack(side=tk.LEFT, padx=5, pady=10)
    btn_salvar = ctk.CTkButton(frame_pesquisa, text="Salvar Alterações", command=salvar_banco_atual, width=150, fg_color="#27ae60", hover_color="#1e8449", font=("Segoe UI", 12, "bold"))
    btn_salvar.pack(side=tk.RIGHT, padx=5, pady=10)

    tv_banco = configurar_treeview(tab_banco)
    tv_banco.tag_configure('cpf_invalido_matematico', background='#f2dede', foreground='black')
    
    def configurar_edicao_treeview(tv, is_banco=True):
        def on_double_click(event):
            region = tv.identify("region", event.x, event.y)
            if region != "cell":
                return
            
            col = tv.identify_column(event.x)
            row_iid = tv.identify_row(event.y)
            if not row_iid:
                return
                
            x, y, width, height = tv.bbox(row_iid, col)
            
            col_idx = int(col[1:]) - 1
            col_name = tv["columns"][col_idx]
            current_value = tv.item(row_iid, "values")[col_idx]
            
            entry = tk.Entry(tv, font=("Segoe UI", 10))
            entry.place(x=x, y=y, width=width, height=height)
            entry.insert(0, current_value if current_value != "---" else "")
            entry.focus()
            
            def save_edit(event=None):
                if not entry.winfo_exists():
                    return
                try:
                    new_value = entry.get()
                    if new_value == "":
                        new_value = "---"
                    
                    if col_name == 'CPF' and new_value not in ["---", "Não identificado", "CPF não identificado", "CPF Inválido", ""]:
                        num_limpo = re.sub(r'\D', '', new_value)
                        if len(num_limpo) == 11:
                            formatted_cpf = f"{num_limpo[:3]}.{num_limpo[3:6]}.{num_limpo[6:9]}-{num_limpo[9:]}"
                            new_value = formatted_cpf
                            
                            if is_banco:
                                if not validar_cpf(formatted_cpf):
                                    messagebox.showwarning(
                                        "CPF Inválido", 
                                        "O CPF digitado é matematicamente inválido (dígito verificador incorreto), mas será salvo.\n"
                                        "A linha será destacada para indicar o problema."
                                    )
                                    tv.item(row_iid, tags=('cpf_invalido_matematico',))
                                else:
                                    tv.item(row_iid, tags=()) # CPF é válido, remove a tag
                    old_value = current_value
                    values = list(tv.item(row_iid, "values"))
                    values[col_idx] = new_value
                    tv.item(row_iid, values=values)
                    idx = int(row_iid)
                    
                    if is_banco:
                        estado_gui["df_banco_atual"].at[idx, col_name] = new_value
                        estado_gui["alterado"] = True
                    else:
                        estado_gui["df_erros_atual"].at[idx, col_name] = new_value
                        
                        row_data = estado_gui["df_erros_atual"].loc[idx].copy()
                        
                        motivo_falha = str(row_data.get('O que falta corrigir?', '---')).strip()
                        if motivo_falha in ['', '---']:
                            estado_gui["df_erros_atual"] = estado_gui["df_erros_atual"].drop(idx)
                            tv.delete(row_iid) # Apaga visualmente da tabela

                        estado_gui["alterado_erros"] = True
                        
                        tipo_novo = row_data.get('Tipo de Termo', 'Não identificado')
                        tipo_velho = old_value if col_name == 'Tipo de Termo' else tipo_novo
                        
                        arq_origem = row_data.get('Arquivo Origem')
                        pagina = row_data.get('Página')
                        
                        dir_base = os.path.dirname(estado_gui["caminho_erros_atual"]) if estado_gui["caminho_erros_atual"] else os.getcwd()
                        mapa_arquivos = {
                            'Cirúrgico': 'BANCO_MESTRE_CIRURGICO.xlsx',
                            'Antiparasitário': 'BANCO_MESTRE_ANTIPARASITARIO.xlsx',
                            'Anestésico': 'BANCO_MESTRE_ANESTESICO.xlsx',
                            'Não identificado': 'BANCO_MESTRE_NAO_IDENTIFICADO.xlsx'
                        }
                        
                        row_dict = row_data.to_dict()
                        if 'O que falta corrigir?' in row_dict:
                            del row_dict['O que falta corrigir?']
                            
                        def atualizar_bd_task(t_velho, t_novo, r_dict, a_origem, pag):
                            def atualizar_bd(tipo, remover=False):
                                nome_b = mapa_arquivos.get(tipo, 'BANCO_MESTRE_NAO_IDENTIFICADO.xlsx')
                                cam_b = os.path.join(dir_base, nome_b)
                                df_b = pd.DataFrame()
                                if os.path.exists(cam_b):
                                    try: df_b = pd.read_excel(cam_b)
                                    except: pass
                                    
                                if not df_b.empty and 'Arquivo Origem' in df_b.columns and 'Página' in df_b.columns:
                                    mask = (df_b['Arquivo Origem'] == a_origem) & (df_b['Página'] == pag)
                                    if remover:
                                        df_b = df_b[~mask]
                                    else:
                                        if mask.any():
                                            for k, v in r_dict.items():
                                                if k in df_b.columns:
                                                    df_b.loc[mask, k] = v
                                        else:
                                            df_b = pd.concat([df_b, pd.DataFrame([r_dict])], ignore_index=True)
                                elif not remover:
                                    df_b = pd.DataFrame([r_dict])
                                    
                                try:
                                    if not df_b.empty:
                                        df_b.to_excel(cam_b, index=False, engine='openpyxl')
                                        if estado_gui["caminho_banco_atual"] == cam_b:
                                            def update_memory(new_df):
                                                estado_gui["df_banco_atual"] = new_df
                                            tv.after(0, update_memory, df_b)
                                except Exception as e_bd:
                                    print(f"Erro ao salvar banco sincronizado: {e_bd}")

                            if t_novo != t_velho:
                                atualizar_bd(t_velho, remover=True)
                                
                            atualizar_bd(t_novo, remover=False)
                            
                        # Roda em thread paralela para não congelar a tela
                        threading.Thread(target=atualizar_bd_task, args=(tipo_velho, tipo_novo, row_dict, arq_origem, pagina), daemon=True).start()
                        
                        # Atualização instantânea (thread-safe) dos cards de indicadores na tela de Erros
                        try:
                            root.after(0, atualizar_cards_erros)
                            root.after(0, atualizar_contadores_tabelas)
                        except Exception:
                            pass
                        
                except Exception as e:
                    print(f"Erro ao salvar edição: {e}")
                finally:
                    entry.destroy()
                
            def cancel_edit(event=None):
                entry.destroy()
                
            entry.bind("<Return>", save_edit)
            entry.bind("<FocusOut>", save_edit)
            entry.bind("<Escape>", cancel_edit)

        tv.bind("<Double-1>", on_double_click)

    configurar_edicao_treeview(tv_banco, is_banco=True)
    
    # --- CARDS DO RELATÓRIO DE ERROS ---
    lbl_instrucao_erros = ctk.CTkLabel(tab_erros, text="💡 Dê um duplo-clique na célula para editar. Ao terminar de corrigir, o registro irá automaticamente para a base concluída.", font=("Segoe UI", 13, "bold"), text_color="#f0ad4e")
    lbl_instrucao_erros.pack(anchor=tk.W, padx=15, pady=(15, 0))
    
    frame_cards_erros = ctk.CTkFrame(tab_erros, fg_color="transparent")
    frame_cards_erros.pack(fill=tk.X, padx=10, pady=(10, 0))

    var_faltantes_tutor = tk.StringVar(value="Tutor não ident.: 0")
    var_faltantes_cpf = tk.StringVar(value="CPF inválido: 0")
    var_faltantes_animal = tk.StringVar(value="Animal não ident.: 0")
    var_faltantes_especie = tk.StringVar(value="Espécie não ident.: 0")
    var_faltantes_tipo = tk.StringVar(value="Tipo não ident.: 0")
    var_faltantes_assinatura = tk.StringVar(value="Sem Assinatura: 0")
    var_repetidos = tk.StringVar(value="Repetidos: 0")

    def atualizar_cards_erros():
        try:
            df_erros = estado_gui["df_erros_atual"]
            if not df_erros.empty:
                var_faltantes_tutor.set(f"Tutor não ident.: {len(df_erros[df_erros['Nome Completo'] == 'Não identificado'])}")
                var_faltantes_cpf.set(f"CPF inválido: {len(df_erros[df_erros['CPF'].isin(['Não identificado', 'CPF não identificado', 'CPF Inválido'])])}")
                var_faltantes_animal.set(f"Animal não ident.: {len(df_erros[df_erros['Nome do Animal'] == 'Não identificado'])}")
                var_faltantes_especie.set(f"Espécie não ident.: {len(df_erros[df_erros['Espécie'] == 'Não identificado'])}")
                var_faltantes_tipo.set(f"Tipo não ident.: {len(df_erros[df_erros['Tipo de Termo'] == 'Não identificado'])}")
                var_faltantes_assinatura.set(f"Sem Assinatura: {len(df_erros[df_erros['Assinatura Presente'] == 'Não'])}")
                if 'O que falta corrigir?' in df_erros.columns:
                    var_repetidos.set(f"Repetidos: {len(df_erros[df_erros['O que falta corrigir?'].astype(str).str.contains('Registro Duplicado')])}")
                else:
                    var_repetidos.set("Repetidos: 0")
            else:
                var_faltantes_tutor.set("Tutor não ident.: 0")
                var_faltantes_cpf.set("CPF inválido: 0")
                var_faltantes_animal.set("Animal não ident.: 0")
                var_faltantes_especie.set("Espécie não ident.: 0")
                var_faltantes_tipo.set("Tipo não ident.: 0")
                var_faltantes_assinatura.set("Sem Assinatura: 0")
                var_repetidos.set("Repetidos: 0")
        except Exception as e:
            print(f"Erro ao atualizar cards de erros: {e}")

    def aplicar_filtro_erros(tipo_filtro):
        df_original = estado_gui["df_erros_atual"]
        if df_original.empty and tipo_filtro is not None:
            return

        if tipo_filtro is None:
            df_filtrado = df_original
        elif tipo_filtro == "tutor":
            df_filtrado = df_original[df_original['Nome Completo'] == 'Não identificado']
        elif tipo_filtro == "cpf":
            df_filtrado = df_original[df_original['CPF'].isin(['Não identificado', 'CPF não identificado', 'CPF Inválido'])]
        elif tipo_filtro == "animal":
            df_filtrado = df_original[df_original['Nome do Animal'] == 'Não identificado']
        elif tipo_filtro == "especie":
            df_filtrado = df_original[df_original['Espécie'] == 'Não identificado']
        elif tipo_filtro == "tipo":
            df_filtrado = df_original[df_original['Tipo de Termo'] == 'Não identificado']
        elif tipo_filtro == "assinatura":
            df_filtrado = df_original[df_original['Assinatura Presente'] == 'Não']
        elif tipo_filtro == "repetidos":
            if 'O que falta corrigir?' in df_original.columns:
                df_filtrado = df_original[df_original['O que falta corrigir?'].astype(str).str.contains('Registro Duplicado')]
            else:
                df_filtrado = df_original.head(0) # Retorna um DataFrame vazio com as mesmas colunas
        else:
            df_filtrado = df_original

        tv = tv_erros
        for row in tv.get_children(): tv.delete(row)
        if df_filtrado.empty: return

        tv["columns"] = list(df_filtrado.columns)
        tv["show"] = "headings"
        for col in tv["columns"]:
            tv.heading(col, text=col)
            tv.column(col, width=150, anchor=tk.CENTER)
        for idx, row in df_filtrado.iterrows():
            tv.insert("", "end", iid=str(idx), values=list(row))
        atualizar_contadores_tabelas()

    def criar_card_erro(parent, text_var, color, filtro):
        card = ctk.CTkFrame(parent, fg_color=color, corner_radius=8, cursor="hand2")
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        card.bind("<Button-1>", lambda e, f=filtro: aplicar_filtro_erros(f))
        lbl = ctk.CTkLabel(card, textvariable=text_var, font=("Segoe UI", 12, "bold"), text_color="white")
        lbl.pack(padx=5, pady=15)
        lbl.bind("<Button-1>", lambda e, f=filtro: aplicar_filtro_erros(f))
        return card

    criar_card_erro(frame_cards_erros, var_faltantes_tutor, "#d9534f", "tutor")
    criar_card_erro(frame_cards_erros, var_faltantes_cpf, "#f0ad4e", "cpf")
    criar_card_erro(frame_cards_erros, var_faltantes_animal, "#5bc0de", "animal")
    criar_card_erro(frame_cards_erros, var_faltantes_especie, "#5cb85c", "especie")
    criar_card_erro(frame_cards_erros, var_faltantes_tipo, "#8e44ad", "tipo")
    criar_card_erro(frame_cards_erros, var_faltantes_assinatura, "#e67e22", "assinatura")
    criar_card_erro(frame_cards_erros, var_repetidos, "#6c757d", "repetidos")

    frame_acoes_erros = ctk.CTkFrame(tab_erros, fg_color="transparent")
    frame_acoes_erros.pack(fill=tk.X, padx=10, pady=(5, 0))

    def excluir_linha_erros():
        selecionados = tv_erros.selection()
        if not selecionados:
            messagebox.showwarning("Aviso", "Selecione uma ou mais linhas na tabela para excluir.")
            return
        if messagebox.askyesno("Confirmar Exclusão", "Deseja realmente excluir a(s) linha(s) selecionada(s) do Relatório de Erros?"):
            indices = [int(item) for item in selecionados]
            estado_gui["df_erros_atual"] = estado_gui["df_erros_atual"].drop(indices).reset_index(drop=True)
            estado_gui["alterado_erros"] = True
            salvar_erros_atual()
            atualizar_tabelas()

    btn_excluir_erros = ctk.CTkButton(frame_acoes_erros, text="Excluir Linha", command=excluir_linha_erros, width=120, fg_color="#d9534f", hover_color="#c9302c")
    btn_excluir_erros.pack(side=tk.LEFT, padx=5, pady=5)

    btn_limpar_filtro_erros = ctk.CTkButton(frame_acoes_erros, text="Limpar Filtro", command=lambda: aplicar_filtro_erros(None), width=120)
    btn_limpar_filtro_erros.pack(side=tk.LEFT, padx=5, pady=5)

    lbl_total_erros = ctk.CTkLabel(frame_acoes_erros, text="Pendências exibidas: 0", font=("Segoe UI", 12, "bold"), text_color="#5bc0de")
    lbl_total_erros.pack(side=tk.RIGHT, padx=15, pady=5)

    tv_erros = configurar_treeview(tab_erros)
    tv_erros.tag_configure('cpf_invalido', background='#f2dede', foreground='black')
    tv_erros.tag_configure('duplicado', background='#fcf8e3', foreground='black')
    tv_erros.tag_configure('corrigido', background='#dff0d8', foreground='black')
    configurar_edicao_treeview(tv_erros, is_banco=False)
    
    def carregar_dados_tv(tv, caminhos_excel, is_banco=False):
        for row in tv.get_children(): tv.delete(row)
        
        if isinstance(caminhos_excel, str):
            caminhos_excel = [caminhos_excel]
            
        df_concat = pd.DataFrame()
        for caminho_excel in caminhos_excel:
            if caminho_excel and os.path.exists(caminho_excel):
                try:
                    df = pd.read_excel(caminho_excel)
                    df_concat = pd.concat([df_concat, df], ignore_index=True)
                except Exception as e:
                    print(f"Erro ao carregar visualização de {caminho_excel}: {e}")

        # Garante que os dados exibidos e salvos estarão sempre ordenados
        if not df_concat.empty and all(col in df_concat.columns for col in ['Arquivo Origem', 'Página']):
            df_concat = df_concat.sort_values(by=['Arquivo Origem', 'Página'], ascending=[True, True]).reset_index(drop=True)

        if not df_concat.empty:
            if 'Motivo da Falha' in df_concat.columns and 'O que falta corrigir?' not in df_concat.columns:
                df_concat = df_concat.rename(columns={'Motivo da Falha': 'O que falta corrigir?'})

            df_original = df_concat.copy()
            df_limpa = limpar_dados_planilha(df_concat)

            # Compara se a limpeza alterou o DataFrame. fillna é crucial para tratar NaNs de forma consistente.
            if not df_original.fillna('temp_nan').equals(df_limpa.fillna('temp_nan')):
                print("ℹ️ Limpeza automática de dados aplicada à planilha carregada.")
                df_concat = df_limpa
                
                # Salva a planilha de volta se for um arquivo único
                if len(caminhos_excel) == 1 and caminhos_excel[0]:
                    try:
                        df_concat.to_excel(caminhos_excel[0], index=False, engine='openpyxl')
                        print(f"💾 Planilha '{os.path.basename(caminhos_excel[0])}' foi atualizada com os dados limpos.")
                    except Exception as e:
                        print(f"❌ Erro ao salvar planilha limpa automaticamente: {e}")

            df_concat = df_concat.fillna("---")
            if is_banco:
                estado_gui["df_banco_atual"] = df_concat
                estado_gui["caminho_banco_atual"] = caminhos_excel[0] if len(caminhos_excel) == 1 else ""
                estado_gui["alterado"] = False
            else:
                estado_gui["df_erros_atual"] = df_concat
                estado_gui["caminho_erros_atual"] = caminhos_excel[0] if len(caminhos_excel) == 1 else ""
                estado_gui["alterado_erros"] = False

            tv["columns"] = list(df_concat.columns)
            tv["show"] = "headings"
            for col in tv["columns"]:
                tv.heading(col, text=col)
                tv.column(col, width=150, anchor=tk.CENTER)
            for idx, row in df_concat.iterrows():
                tags = ()
                if is_banco:
                    cpf_val = str(row.get('CPF', ''))
                    if cpf_val not in ["---", "Não identificado", "CPF não identificado", "CPF Inválido", ""] and not validar_cpf(cpf_val):
                        tags = ('cpf_invalido_matematico',)
                else:
                    temp_tags = []
                    motivo = str(row.get('O que falta corrigir?', ''))
                    if 'CPF' in motivo:
                        temp_tags.append('cpf_invalido')
                    if 'Duplicado' in motivo:
                        temp_tags.append('duplicado')
                    if '--- Corrigido ---' in motivo:
                        temp_tags.append('corrigido')
                    tags = tuple(temp_tags)
                
                tv.insert("", "end", iid=str(idx), values=list(row), tags=tags)


        else:
            if is_banco:
                estado_gui["df_banco_atual"] = pd.DataFrame()
                estado_gui["caminho_banco_atual"] = ""
                estado_gui["alterado"] = False
            else:
                estado_gui["df_erros_atual"] = pd.DataFrame()
                estado_gui["caminho_erros_atual"] = ""
                estado_gui["alterado_erros"] = False

    def atualizar_contadores_tabelas():
        try:
            lbl_total_banco.configure(text=f"Registros: {len(tv_banco.get_children())}")
        except Exception:
            pass
        try:
            lbl_total_erros.configure(text=f"Pendências exibidas: {len(tv_erros.get_children())}")
        except Exception:
            pass

    def atualizar_tabelas():
        if estado_gui.get("alterado"):
            salvar_banco_atual()
        if estado_gui.get("alterado_erros"):
            salvar_erros_atual()
            
        caminho_input = entrada_pdf.get()
        if caminho_input and os.path.exists(caminho_input):
            if os.path.isdir(caminho_input):
                diretorio_base = caminho_input
            else:
                diretorio_base = os.path.dirname(caminho_input)
        else:
            diretorio_base = os.getcwd()
            
        mapa_arquivos = {
            'Cirúrgico': 'BANCO_MESTRE_CIRURGICO.xlsx',
            'Antiparasitário': 'BANCO_MESTRE_ANTIPARASITARIO.xlsx',
            'Anestésico': 'BANCO_MESTRE_ANESTESICO.xlsx',
            'Não identificado': 'BANCO_MESTRE_NAO_IDENTIFICADO.xlsx'
        }
        tipo_selecionado = tipo_banco_var.get()
        banco_mestre_path = os.path.join(diretorio_base, mapa_arquivos.get(tipo_selecionado, 'BANCO_MESTRE_CIRURGICO.xlsx'))
        revisao_manual_path = os.path.join(diretorio_base, 'REVISAO_MANUAL_CASTRA.xlsx')
        
        carregar_dados_tv(tv_banco, banco_mestre_path, is_banco=True)
        carregar_dados_tv(tv_erros, revisao_manual_path, is_banco=False)
        
        # --- Atualizar Cards ---
        atualizar_cards_erros()
        atualizar_contadores_tabelas()

    # Tenta carregar tabelas do diretório atual se existirem
    autosave()
    atualizar_tabelas()
    root.mainloop()

if __name__ == "__main__":
    iniciar_gui()
