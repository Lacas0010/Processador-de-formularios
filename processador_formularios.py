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


def converter_pdf_para_imagens(caminho_pdf: str, dpi: int) -> list:
    """
    Converte todas as páginas de um arquivo PDF em uma lista de imagens PIL.

    Args:
        caminho_pdf (str): O caminho para o arquivo PDF.
        dpi (int): A resolução das imagens a serem geradas.

    Returns:
        list: Uma lista de objetos de imagem PIL.
    """
    nome_arquivo = os.path.basename(caminho_pdf)
    print(f"📂 ARQUIVO: {nome_arquivo}")

    try:
        pdf = pdfium.PdfDocument(caminho_pdf)
        imagens = []
        
        # O pypdfium2 usa 72 DPI como base padrão, calculamos a escala para 300 DPI
        escala = dpi / 72.0

        for i in range(len(pdf)):
            pagina = pdf[i]
            bitmap = pagina.render(scale=escala)
            img = bitmap.to_pil()
            imagens.append(img)
            
        print(f"📑 Total de páginas identificadas: {len(imagens)}\n")
        return imagens
    except Exception as e:
        print(f"❌ Erro ao ler o arquivo: {e}")
        return []


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
    metade_y = int(altura_imagem * 0.5)

    dados_ocr = pytesseract.image_to_data(imagem_processada, output_type=pytesseract.Output.DICT, lang='por')
    n_boxes = len(dados_ocr['text'])
    
    # Múltiplas âncoras para não depender de ler apenas a palavra "assinatura"
    ancoras = ['assinatura', 'assina', 'tutor', 'proprietário', 'proprietario', 'responsável', 'responsavel']
    
    for i in range(n_boxes):
        y = dados_ocr['top'][i]
        
        if y < metade_y:
            continue

        texto_box = dados_ocr['text'][i].strip().lower()
        
        if any(ancora in texto_box for ancora in ancoras):
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

            pixels_brancos = cv2.countNonZero(area_assinatura)
            pixels_pretos = total_pixels - pixels_brancos
            densidade_tinta = pixels_pretos / total_pixels
            
            if densidade_tinta > 0.015:
                return True
                
    # --- FALLBACK GEOMÉTRICO ---
    # Se o Tesseract não ler nenhuma âncora, recorta os 15% inferiores e 50% direitos da página
    corte_y_fallback = int(altura_imagem * 0.85)
    corte_x_fallback = int(largura_imagem * 0.50)
    
    area_fallback = imagem_processada[corte_y_fallback:altura_imagem, corte_x_fallback:largura_imagem]
    total_pixels_fall = area_fallback.size
    
    if total_pixels_fall > 0:
        pixels_brancos_fall = cv2.countNonZero(area_fallback)
        pixels_pretos_fall = total_pixels_fall - pixels_brancos_fall
        densidade_fall = pixels_pretos_fall / total_pixels_fall
        
        if densidade_fall > 0.015:
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
        r'abaixo\s*aser', r'\bpala\b', r'\bpero\b'
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
    """Extração cirúrgica baseada na estrutura real do formulário SEPDA."""
    dados = {
        "Nome Completo": "Não identificado",
        "CPF": "CPF não identificado",
        "Nome do Animal": "Não identificado",
        "Espécie": "Não identificado"
    }

    # Transforma o texto em lista de linhas para facilitar a busca por vizinhança
    linhas = [l.strip() for l in texto.split('\n') if l.strip()]
    
    # 1. BUSCA DO CPF (Âncora Principal)
    cpf_linha_idx = -1
    for i, linha in enumerate(linhas):
        # Filtro Anti-Falsos Positivos: Evita concatenar números de endereço/telefone que coincidam com um CPF válido
        if re.search(r'(?i)\b(?:cep|quadra|lote|conjunto|casa|setor|rua|avenida|residente|bairro|endere[çc]o|tel|telefone|celular|cel)\b', linha):
            # Só ignora a linha se ela NÃO tiver a palavra 'CPF' explícita
            if not re.search(r'(?i)\bcpf\b', linha):
                continue
                
        num_limpo = re.sub(r'\D', '', linha)
        if len(num_limpo) >= 11:
            # Testa janelas de 11 dígitos iterativamente para tolerar caracteres extras lidos pelo OCR
            encontrou_valido = False
            for j in range(len(num_limpo) - 10):
                num_cpf = num_limpo[j:j+11]
                if validar_cpf(num_cpf):
                    dados["CPF"] = f"{num_cpf[:3]}.{num_cpf[3:6]}.{num_cpf[6:9]}-{num_cpf[9:]}"
                    encontrou_valido = True
                    cpf_linha_idx = i
                    break
            
            if encontrou_valido:
                break
            elif dados["CPF"] == "CPF não identificado" and len(num_limpo) == 11:
                dados["CPF"] = "CPF Inválido"
                cpf_linha_idx = i
                print(f"  ⚠️ Alerta: Ignorando numeração semelhante a CPF (inválida): {num_limpo[:11]}")

    # 2. BUSCA DO NOME (Tutor)
    for i, linha in enumerate(linhas):
        # Ignora linhas de rodapé, observações, carimbos médicos ou "assinatura"
        if re.search(r'(?i)(?:observaç[õo]es|impresso|p[áa]g|assinatura|pelo\s+animal|crmv)', linha):
            continue
            
        # Filtro Anti-Animal: Impede que a linha com os dados do pet seja lida como tutor se começar apenas com "Nome:"
        if re.search(r'(?i)\b(?:felin[oa]|canin[oa]|f[êe]mea|macho|nascid[oa]|pelagem|srd|kg|chip)\b', linha):
            continue
            
        # Busca RIGOROSA pelo tutor (Adicionado 'nome\b' para capturar rótulos simples no topo)
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

    # Fallback Imbatível do Nome: Se o rótulo falhou, captura a linha diretamente acima do CPF.
    # IMPORTANTE: Só ativa o fallback se o CPF for VÁLIDO. Evita âncoras falsas de telefones/CEPs.
    if dados["Nome Completo"] == "Não identificado" and cpf_linha_idx > 0 and dados["CPF"] not in ["CPF Inválido", "CPF não identificado"]:
        for j in range(cpf_linha_idx - 1, max(-1, cpf_linha_idx - 4), -1):
            candidato = limpar_valor(linhas[j])
            # Evita capturar RGs soltos
            if candidato != "Não identificado" and not re.search(r'(?i)(?:cpf|rg|doc)', linhas[j]):
                dados["Nome Completo"] = candidato
                break

    # 3. BUSCA DO ANIMAL E ESPÉCIE 
    # Bloco Independente com Âncora à Prova de Balas
    for i, linha in enumerate(linhas):
        # Gatilho Flexível: Ativa por "identificação" ou diretamente pela palavra "espécie"
        if re.search(r'(?i)(?:identifica[çc][ãa]o|dados\s+do\s+animal|paciente|esp[eéè]cie)', linha):
            # Olha a linha atual e as 6 seguintes
            busca_area = " \n ".join(linhas[i:i+7]) 
            
            # Imunidade a "Identificação do Proprietário": se ativou por identificação mas não tem palavras de pet, ignora
            if "identifica" in linha.lower() and not re.search(r'(?i)(?:esp[eéè]cie|ra[çc]a|pelagem|sexo|idade|f[êe]mea|macho)', busca_area):
                continue
            
            # Extrai Animal: Lookahead negativo (?!\s+completo) impede de engolir a linha do tutor
            match_animal = re.search(r'(?i)\bnome\b(?!\s+completo)[\s:\-.]*([A-Za-zÀ-ú\s]+?)(?:[\n,]|esp[eéè]cie|ra[çc]a|sexo|idade|f[êe]mea|macho|nascid[oa]|pelagem)', busca_area)
            if match_animal:
                dados["Nome do Animal"] = limpar_valor(match_animal.group(1))
                
            # Extrai Espécie com os mesmos bloqueios
            match_especie = re.search(r'(?i)esp[eéè]cie[\s:\-.]*([A-Za-zÀ-ú\s]+?)(?:[\n,]|$|ra[çc]a|sexo|idade|f[êe]mea|macho|nascid[oa]|pelagem)', busca_area)
            if match_especie:
                dados["Espécie"] = limpar_valor(match_especie.group(1))
            
            if dados["Nome do Animal"] != "Não identificado" or dados["Espécie"] != "Não identificado":
                break
            
    # 4. FALLBACK PLANO B: ESPÉCIE
    # Busca a Espécie isoladamente em qualquer linha, caso tenha falhado
    if dados["Espécie"] == "Não identificado":
        for linha in linhas:
            match_especie_fallback = re.search(r'(?i)esp[eéè]cie\s*[:\-]?\s*([A-Za-zÀ-ú\s]+)', linha)
            if match_especie_fallback:
                especie_limpa = limpar_valor(match_especie_fallback.group(1))
                if especie_limpa != "Não identificado":
                    dados["Espécie"] = especie_limpa
                    break
                    
    # --- VALIDAÇÃO FINAL ANTI-CROSS-CAPTURE ---
    # Se o nome do animal for idêntico ao do tutor (ou englobá-lo), descarta o animal para evitar duplicação irreal
    if dados["Nome Completo"] != "Não identificado" and dados["Nome do Animal"] == dados["Nome Completo"]:
        print(f"  ⚠️ Alerta: Pet e Tutor com mesmo nome. Omitindo pet (possível falso positivo).")
        dados["Nome do Animal"] = "Não identificado"

    # --- PADRONIZAÇÃO DE TEXTO (TITLE CASE) ---
    # Formata os nomes para ficarem bonitos no Excel (ex: 'leo' -> 'Leo', 'MALU' -> 'Malu')
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
    imagens_pil = converter_pdf_para_imagens(caminho_pdf, DPI_CONVERSAO)
    if not imagens_pil:
        return

    dados_coletados = []
    total_paginas = len(imagens_pil)
    
    for i, imagem_pil in enumerate(imagens_pil):
        print(f"📄 PÁGINA {i+1} / {total_paginas}")
        print("━" * 45)
        if progresso_callback:
            progresso_callback(i, total_paginas)

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

        dados_pagina["Página"] = i + 1

        # 5. Assinatura
        assinatura_presente = verificar_assinatura(imagem_processada)
        dados_pagina["Assinatura Presente"] = "Sim" if assinatura_presente else "Não"

        status_ass = "✅ Identificada" if assinatura_presente else "❌ Não identificada"
        print(f"  👤 Tutor:      {dados_pagina.get('Nome Completo', '')}")
        print(f"  🪪 CPF:        {dados_pagina.get('CPF', '')}")
        print(f"  🐾 Pet:        {dados_pagina.get('Nome do Animal', '')} ({dados_pagina.get('Espécie', '')})")
        print(f"  ✍️ Assinatura: {status_ass}")
        print("━" * 45 + "\n")
        dados_coletados.append(dados_pagina)

    if progresso_callback:
        progresso_callback(total_paginas, total_paginas)

    if not dados_coletados:
        return

    # --- INTELIGÊNCIA DE BANCO DE DADOS LOCAL ---
    print("💾 Sincronizando dados com as planilhas do sistema...")
    df_novo = pd.DataFrame(dados_coletados)
    
    diretorio_base = os.path.dirname(caminho_pdf)
    nome_arquivo = os.path.basename(caminho_pdf)
    banco_mestre_path = os.path.join(diretorio_base, 'BANCO_MESTRE_CASTRA.xlsx')
    revisao_manual_path = os.path.join(diretorio_base, 'REVISAO_MANUAL_CASTRA.xlsx')

    # 1. Atualizar Banco Mestre (Acumulado)
    df_mestre_check = pd.DataFrame()
    if os.path.exists(banco_mestre_path):
        try:
            df_mestre = pd.read_excel(banco_mestre_path)
            df_mestre_check = df_mestre.copy()
            df_final = pd.concat([df_mestre, df_novo], ignore_index=True)
        except Exception:
            df_final = df_novo.copy()
    else:
        df_final = df_novo.copy()

    # Remove Duplicados (CPF e Nome do Animal), protegendo os não identificados contra deleção em massa
    mascara_validos = (df_final['CPF'] != 'CPF não identificado') & (df_final['CPF'] != 'Não identificado') & (df_final['CPF'] != 'CPF Inválido')
    duplicados = df_final[mascara_validos].duplicated(subset=['CPF', 'Nome do Animal'], keep='last')
    df_final = df_final.drop(duplicados[duplicados].index)
    
    colunas_ordem = ['Página', 'Nome Completo', 'CPF', 'Nome do Animal', 'Espécie', 'Assinatura Presente']
    df_final = df_final[[col for col in colunas_ordem if col in df_final.columns]]
    
    try:
        df_final.to_excel(banco_mestre_path, index=False, engine='openpyxl')
        print(f"✅ Base acumulada atualizada: {banco_mestre_path}")
    except Exception as e:
        print(f"❌ Erro ao salvar base acumulada: {e}")

    # 2. Relatório de Inconsistências (Revisão Humana)
    erros_list = []
    vistos_lote = set()
    for _, row in df_novo.iterrows():
        motivos = []
        if row.get('Nome Completo') == "Não identificado":
            motivos.append("Nome Ausente/Ilegível")
        if row.get('CPF') in ["CPF não identificado", "CPF Inválido", "Não identificado"]:
            motivos.append("CPF Ausente/Inválido")
        if row.get('Nome do Animal') == "Não identificado":
            motivos.append("Animal Ausente/Ilegível")
        if row.get('Espécie') == "Não identificado":
            motivos.append("Espécie Ausente/Ilegível")
        if row.get('Assinatura Presente') == "Não":
            motivos.append("Assinatura Pendente")
        
        # Verifica duplicidade (cruzando com banco mestre existente e com o lote atual)
        is_dup = False
        cpf_valido = row.get('CPF') not in ["CPF não identificado", "CPF Inválido", "Não identificado"]
        if cpf_valido:
            cpf = row.get('CPF')
            animal = row.get('Nome do Animal')
            if not df_mestre_check.empty and 'CPF' in df_mestre_check.columns and 'Nome do Animal' in df_mestre_check.columns:
                match = (df_mestre_check['CPF'] == cpf) & (df_mestre_check['Nome do Animal'] == animal)
                if match.any():
                    is_dup = True
            
            if (cpf, animal) in vistos_lote:
                is_dup = True
            else:
                vistos_lote.add((cpf, animal))

        if is_dup:
            motivos.append("Registro Duplicado")

        if motivos:
            dict_erro = row.to_dict()
            dict_erro['Motivo da Falha'] = " | ".join(motivos)
            dict_erro['Arquivo Origem'] = nome_arquivo
            erros_list.append(dict_erro)

    if itens_revisao_callback:
        itens_revisao_callback(len(erros_list))

    if erros_list:
        df_erros = pd.DataFrame(erros_list)
        if os.path.exists(revisao_manual_path):
            try:
                df_revisao_antigo = pd.read_excel(revisao_manual_path)
                df_revisao_final = pd.concat([df_revisao_antigo, df_erros], ignore_index=True)
            except Exception:
                df_revisao_final = df_erros
        else:
            df_revisao_final = df_erros
        
        # Deduplicação do relatório de erros
        try:
            df_revisao_final = df_revisao_final.drop_duplicates(subset=['CPF', 'Nome do Animal', 'Motivo da Falha', 'Arquivo Origem'], keep='last')
        except Exception:
            pass
        
        try:
            df_revisao_final.to_excel(revisao_manual_path, index=False, engine='openpyxl')
            print(f"⚠️ Relatório de revisão gerado: {revisao_manual_path}")
        except Exception as e:
            print(f"❌ Erro ao salvar relatório de revisão: {e}")


def iniciar_gui():
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    root = ctk.CTk()
    root.title("Castra-DF - Processador de Formulários Mestre")
    root.geometry("1100x750")
    
    style = ttk.Style(root)
    style.theme_use('clam')
    
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
    
    notebook = ctk.CTkTabview(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
    
    tab_processar = notebook.add('⚙️ Processar PDFs')
    tab_banco = notebook.add('🗄️ Base de Dados Acumulada')
    tab_erros = notebook.add('⚠️ Relatório de Erros')
    
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
    
    btn_processar = ctk.CTkButton(tab_processar, text="▶ INICIAR", font=("Segoe UI", 14, "bold"), height=40, command=lambda: iniciar_thread())
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
                    
                root.after(0, lambda: messagebox.showinfo("Concluído", "Processamento finalizado!\nVerifique as abas de Banco e Erros."))
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
    
    ctk.CTkLabel(frame_pesquisa, text="Pesquisar por Nome/CPF:", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=(10, 5), pady=10)
    entrada_pesquisa = ctk.CTkEntry(frame_pesquisa, font=("Segoe UI", 12), width=250)
    entrada_pesquisa.pack(side=tk.LEFT, padx=5, pady=10)
    
    def pesquisar_banco():
        termo = entrada_pesquisa.get().strip().lower()
        if not termo:
            return
            
        caminho_input = entrada_pdf.get()
        if caminho_input and os.path.exists(caminho_input):
            if os.path.isdir(caminho_input):
                diretorio_base = caminho_input
            else:
                diretorio_base = os.path.dirname(caminho_input)
        else:
            diretorio_base = os.getcwd()
            
        banco_mestre_path = os.path.join(diretorio_base, 'BANCO_MESTRE_CASTRA.xlsx')
        if not os.path.exists(banco_mestre_path):
            return
            
        try:
            df = pd.read_excel(banco_mestre_path).fillna("---")
            mask = df['Nome Completo'].astype(str).str.lower().str.contains(termo) | \
                   df['CPF'].astype(str).str.lower().str.contains(termo)
            df_filtrado = df[mask]
            
            for row in tv_banco.get_children(): tv_banco.delete(row)
            tv_banco["columns"] = list(df_filtrado.columns)
            tv_banco["show"] = "headings"
            for col in tv_banco["columns"]:
                tv_banco.heading(col, text=col)
                tv_banco.column(col, width=150, anchor=tk.CENTER)
            for _, row in df_filtrado.iterrows():
                tv_banco.insert("", "end", values=list(row))
        except Exception as e:
            print(f"Erro ao pesquisar: {e}")

    def limpar_filtro():
        entrada_pesquisa.delete(0, "end")
        atualizar_tabelas()

    ctk.CTkButton(frame_pesquisa, text="Pesquisar", command=pesquisar_banco, width=100).pack(side=tk.LEFT, padx=5, pady=10)
    ctk.CTkButton(frame_pesquisa, text="Limpar Filtro", command=limpar_filtro, width=100).pack(side=tk.LEFT, padx=5, pady=10)

    tv_banco = configurar_treeview(tab_banco)
    
    # --- CARDS DO RELATÓRIO DE ERROS ---
    frame_cards_erros = ctk.CTkFrame(tab_erros, fg_color="transparent")
    frame_cards_erros.pack(fill=tk.X, padx=10, pady=(10, 0))

    var_faltantes_tutor = tk.StringVar(value="Tutor não ident.: 0")
    var_faltantes_cpf = tk.StringVar(value="CPF inválido: 0")
    var_faltantes_animal = tk.StringVar(value="Animal não ident.: 0")
    var_faltantes_especie = tk.StringVar(value="Espécie não ident.: 0")
    var_repetidos = tk.StringVar(value="Repetidos: 0")

    def criar_card_erro(parent, text_var, color):
        card = ctk.CTkFrame(parent, fg_color=color, corner_radius=8)
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        lbl = ctk.CTkLabel(card, textvariable=text_var, font=("Segoe UI", 12, "bold"), text_color="white")
        lbl.pack(padx=5, pady=15)
        return card

    criar_card_erro(frame_cards_erros, var_faltantes_tutor, "#d9534f")   # Vermelho
    criar_card_erro(frame_cards_erros, var_faltantes_cpf, "#f0ad4e")     # Laranja
    criar_card_erro(frame_cards_erros, var_faltantes_animal, "#5bc0de")  # Azul Claro
    criar_card_erro(frame_cards_erros, var_faltantes_especie, "#5cb85c") # Verde
    criar_card_erro(frame_cards_erros, var_repetidos, "#6c757d")         # Cinza

    tv_erros = configurar_treeview(tab_erros)
    
    def carregar_dados_tv(tv, caminho_excel):
        for row in tv.get_children(): tv.delete(row)
        if os.path.exists(caminho_excel):
            try:
                df = pd.read_excel(caminho_excel).fillna("---")
                tv["columns"] = list(df.columns)
                tv["show"] = "headings"
                for col in tv["columns"]:
                    tv.heading(col, text=col)
                    tv.column(col, width=150, anchor=tk.CENTER)
                for _, row in df.iterrows():
                    tv.insert("", "end", values=list(row))
            except Exception as e:
                print(f"Erro ao carregar visualização de {caminho_excel}: {e}")

    def atualizar_tabelas():
        caminho_input = entrada_pdf.get()
        if caminho_input and os.path.exists(caminho_input):
            if os.path.isdir(caminho_input):
                diretorio_base = caminho_input
            else:
                diretorio_base = os.path.dirname(caminho_input)
        else:
            diretorio_base = os.getcwd()
            
        banco_mestre_path = os.path.join(diretorio_base, 'BANCO_MESTRE_CASTRA.xlsx')
        revisao_manual_path = os.path.join(diretorio_base, 'REVISAO_MANUAL_CASTRA.xlsx')
        
        carregar_dados_tv(tv_banco, banco_mestre_path)
        carregar_dados_tv(tv_erros, revisao_manual_path)
        
        # --- Atualizar Cards ---
        if os.path.exists(revisao_manual_path):
            try:
                df_erros = pd.read_excel(revisao_manual_path)
                
                tutor_ausente = len(df_erros[df_erros['Nome Completo'] == 'Não identificado'])
                cpf_ausente = len(df_erros[df_erros['CPF'].isin(['Não identificado', 'CPF não identificado', 'CPF Inválido'])])
                animal_ausente = len(df_erros[df_erros['Nome do Animal'] == 'Não identificado'])
                especie_ausente = len(df_erros[df_erros['Espécie'] == 'Não identificado'])
                repetidos = len(df_erros[df_erros['Motivo da Falha'].astype(str).str.contains('Registro Duplicado')])
                
                var_faltantes_tutor.set(f"Tutor não ident.: {tutor_ausente}")
                var_faltantes_cpf.set(f"CPF inválido: {cpf_ausente}")
                var_faltantes_animal.set(f"Animal não ident.: {animal_ausente}")
                var_faltantes_especie.set(f"Espécie não ident.: {especie_ausente}")
                var_repetidos.set(f"Repetidos: {repetidos}")
            except Exception as e:
                print(f"Erro ao atualizar cards: {e}")
        else:
            var_faltantes_tutor.set("Tutor não ident.: 0")
            var_faltantes_cpf.set("CPF inválido: 0")
            var_faltantes_animal.set("Animal não ident.: 0")
            var_faltantes_especie.set("Espécie não ident.: 0")
            var_repetidos.set("Repetidos: 0")

    # Tenta carregar tabelas do diretório atual se existirem
    atualizar_tabelas()
    root.mainloop()

if __name__ == "__main__":
    iniciar_gui()
