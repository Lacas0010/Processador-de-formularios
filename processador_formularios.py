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
from tkinter import filedialog, messagebox
import threading

# --- CONFIGURAÇÕES GLOBAIS ---

# Para usuários de Windows, especifique o caminho para o executável do Tesseract.
# Se o Tesseract não estiver no PATH do sistema, descomente e ajuste a linha abaixo.
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\07049770108\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

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
    print(f"Convertendo PDF '{caminho_pdf}' para imagens com {dpi} DPI usando pypdfium2...")

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
            
        print(f"Conversão concluída. {len(imagens)} páginas encontradas.")
        return imagens
    except Exception as e:
        print(f"Erro ao converter PDF: {e}")
        return []


def pre_processar_imagem_para_ocr(imagem_pil: Image) -> np.ndarray:
    """
    Aplica pré-processamento em uma imagem para melhorar a precisão do OCR.
    Converte para escala de cinza e aplica um thresholding adaptativo.

    Args:
        imagem_pil (Image): A imagem no formato PIL.

    Returns:
        np.ndarray: A imagem processada no formato OpenCV (numpy array).
    """
    # Converte de PIL Image para formato OpenCV (Numpy Array)
    imagem_cv = np.array(imagem_pil)
    # Converte para escala de cinza
    cinza = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2GRAY)
    
    # Suavização com GaussianBlur para remover ruídos comuns de digitalização (poeira, texturas)
    desfoque = cv2.GaussianBlur(cinza, (5, 5), 0)
    
    # Aplica binarização adaptativa, excelente para lidar com sombras e iluminação irregular em scans
    processada = cv2.adaptiveThreshold(
        desfoque, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Operação morfológica de abertura para eliminar pequenos pontos pretos (ruído).
    # Invertemos a imagem pois o MORPH_OPEN atua reduzindo os pixels brancos (fundo).
    kernel = np.ones((2, 2), np.uint8)
    inv = cv2.bitwise_not(processada)
    inv_aberta = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
    processada = cv2.bitwise_not(inv_aberta)
    
    return processada


def verificar_assinatura(imagem_processada: np.ndarray) -> bool:
    """
    Localiza dinamicamente a área de assinatura via OCR e verifica se está preenchida.

    Args:
        imagem_processada (np.ndarray): A imagem binarizada.

    Returns:
        bool: True se a assinatura for detectada, False caso contrário.
    """
    # Define o limite para a metade inferior da página
    altura_imagem, _ = imagem_processada.shape
    metade_y = int(altura_imagem * 0.5)

    # Extrai dados do OCR (palavras e suas coordenadas)
    dados_ocr = pytesseract.image_to_data(imagem_processada, output_type=pytesseract.Output.DICT, lang='por')
    
    n_boxes = len(dados_ocr['text'])
    for i in range(n_boxes):
        y = dados_ocr['top'][i]
        
        # Aprimoramento da Assinatura: ignora palavras que estão na metade superior
        if y < metade_y:
            continue

        texto_box = dados_ocr['text'][i].strip().lower()
        
        # Tolerância para achar a palavra (pode ter ruído do OCR)
        if 'assinatura' in texto_box or 'assina' in texto_box:
            x, y = dados_ocr['left'][i], dados_ocr['top'][i]
            w, h = dados_ocr['width'][i], dados_ocr['height'][i]
            
            # Define ROI dinâmico: EXATAMENTE acima da linha de assinatura
            roi_h = 150  # Altura projetada da área de assinatura
            roi_y = max(0, y - roi_h - 20)  # Margem ampliada para ignorar a linha impressa
            roi_w = max(w * 3, 200) # Garante largura mínima cobrindo a extensão da linha
            roi_x = max(0, x - int((roi_w - w) / 2)) # Centraliza o ROI em relação à palavra
            
            # Recorta a área da assinatura da imagem
            area_assinatura = imagem_processada[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

            total_pixels = area_assinatura.size
            if total_pixels == 0:
                continue

            # Calcula a densidade de pixels pretos (tinta) em imagens com limiarização adaptativa
            pixels_brancos = cv2.countNonZero(area_assinatura)
            pixels_pretos = total_pixels - pixels_brancos

            densidade_tinta = pixels_pretos / total_pixels
            print(f"    [LOG] Região de Assinatura mapeada. Densidade de tinta: {densidade_tinta:.4f}")

            # Consideramos assinado se tiver mais de 3% da área com tinta preta (evita sujeiras do scan)
            if densidade_tinta > 0.03:
                return True
            
    return False


def limpar_valor(texto: str) -> str:
    """
    Remove rótulos fantasmas capturados pelo OCR e limpa caracteres especiais, 
    garantindo que fique apenas o valor limpo extraído.
    """
    if not texto:
        return "Não identificado"
    
    termos_remover = [
        r'nome completo', r'\(legível\)', r'assinatura', 
        r'responsável pelo animal', r'responsável', r'espécie', 
        r'especie', r'animal', r'cpf', r'paciente', r'data', r'rg',
        r'tutor', r'proprietário', r'nome do cliente', r'raça', r'idade',
        r'cor', r'sexo', r'peso', r'pelagem', r'tipo', r'nome', r'dono',
        r'observações de interesse', r'relatado verbalmente', r'aproprietarioaresponsavel',
        r'identificação do animal', r'\bpelo\b', r'impresso.*', r'p[áa]g\w*', r'crmv.*',
        r'dados do animal', r'doc\.*', r'identifica[çc][ãa]o'
    ]
    padrao = r'(?i)\b(?:' + '|'.join(termos_remover) + r')\b'
    texto_limpo = re.sub(padrao, '', str(texto))
    
    # Remove pontuações ruidosas do OCR mantendo apenas letras, acentos e espaços
    texto_limpo = re.sub(r'[^A-Za-zÀ-ú ]', '', texto_limpo)
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()
    
    # Limpeza Inteligente: Remove letras soltas (ruído residual de OCR) no início, meio ou fim
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
                    print(f"    [LOG] CPF matemático válido ancorado na linha {i}: {dados['CPF']}")
                    break
            
            if encontrou_valido:
                break
            elif dados["CPF"] == "CPF não identificado" and len(num_limpo) == 11:
                dados["CPF"] = "CPF Inválido"
                cpf_linha_idx = i
                print(f"    [ALERTA] CPF com 11 dígitos capturado, mas reprovado no Dígito Verificador: '{num_limpo[:11]}'")

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
                print(f"    [LOG] Tutor encontrado na linha do rótulo: '{nome_limpo}'")
                break
            elif i + 1 < len(linhas):
                nome_limpo_abaixo = limpar_valor(linhas[i+1])
                if nome_limpo_abaixo != "Não identificado":
                    dados["Nome Completo"] = nome_limpo_abaixo
                    print(f"    [LOG] Tutor encontrado na linha inferior ao rótulo: '{nome_limpo_abaixo}'")
                    break

    # Fallback Imbatível do Nome: Se o rótulo falhou, captura a linha diretamente acima do CPF.
    # IMPORTANTE: Só ativa o fallback se o CPF for VÁLIDO. Evita âncoras falsas de telefones/CEPs.
    if dados["Nome Completo"] == "Não identificado" and cpf_linha_idx > 0 and dados["CPF"] not in ["CPF Inválido", "CPF não identificado"]:
        for j in range(cpf_linha_idx - 1, max(-1, cpf_linha_idx - 4), -1):
            candidato = limpar_valor(linhas[j])
            # Evita capturar RGs soltos
            if candidato != "Não identificado" and not re.search(r'(?i)(?:cpf|rg|doc)', linhas[j]):
                dados["Nome Completo"] = candidato
                print(f"    [LOG] Tutor recuperado via fallback (acima do CPF): '{candidato}'")
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
                if dados["Nome do Animal"] != "Não identificado":
                    print(f"    [LOG] Nome do Animal extraído: '{dados['Nome do Animal']}'")
                
            # Extrai Espécie com os mesmos bloqueios
            match_especie = re.search(r'(?i)esp[eéè]cie[\s:\-.]*([A-Za-zÀ-ú\s]+?)(?:[\n,]|$|ra[çc]a|sexo|idade|f[êe]mea|macho|nascid[oa]|pelagem)', busca_area)
            if match_especie:
                dados["Espécie"] = limpar_valor(match_especie.group(1))
                if dados["Espécie"] != "Não identificado":
                    print(f"    [LOG] Espécie extraída: '{dados['Espécie']}'")
            
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
                    print(f"    [LOG] Espécie recuperada via fallback de varredura: '{especie_limpa}'")
                    break
                    
    # --- VALIDAÇÃO FINAL ANTI-CROSS-CAPTURE ---
    # Se o nome do animal for idêntico ao do tutor (ou englobá-lo), descarta o animal para evitar duplicação irreal
    if dados["Nome Completo"] != "Não identificado" and dados["Nome do Animal"] == dados["Nome Completo"]:
        print(f"    [ALERTA] Falso Positivo interceptado: Nome do Animal ('{dados['Nome do Animal']}') foi anulado por ser espelho do Tutor.")
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
        self.widget_texto.insert(tk.END, string)
        self.widget_texto.see(tk.END) # Faz o scroll automático ir para baixo

    def flush(self):
        pass


def processar_pdf_formularios(caminho_pdf: str, arquivo_saida: str = "resultado_formularios.xlsx"):
    """
    Função principal que orquestra todo o processo de extração de dados do PDF.
    """
    imagens_pil = converter_pdf_para_imagens(caminho_pdf, DPI_CONVERSAO)
    if not imagens_pil:
        return

    dados_coletados = []
    for i, imagem_pil in enumerate(imagens_pil):
        print(f"\nProcessando página {i+1}/{len(imagens_pil)}...")

        # 1. Pré-processamento da imagem
        imagem_processada = pre_processar_imagem_para_ocr(imagem_pil)

        # Filtro de Área: Ignora os primeiros 5% da página (cabeçalhos médicos)
        h_img, w_img = imagem_processada.shape
        corte_y = int(h_img * 0.05)
        imagem_dados = imagem_processada[corte_y:h_img, 0:w_img]

        # 2. Extração de texto com OCR (apenas na área recortada de dados)
        print("  - Executando OCR na área de dados (inferior 95%)...")
        # lang='por' para usar o modelo de linguagem em português
        texto_extraido = pytesseract.image_to_string(
            imagem_dados, lang='por')

        # 3. Parseamento dos dados do texto
        print("  - Extraindo campos específicos...")
        dados_pagina = parsear_dados_do_texto(texto_extraido)
        dados_pagina["Página"] = i + 1

        # 4. Verificação da assinatura
        print("  - Verificando assinatura dinamicamente...")
        assinatura_presente = verificar_assinatura(imagem_processada)
        dados_pagina["Assinatura Presente"] = assinatura_presente

        print(f"  - Dados extraídos: {dados_pagina}")
        dados_coletados.append(dados_pagina)

    # 5. Criação do DataFrame e análise de duplicidade
    print("\nCriando DataFrame e analisando duplicidades...")
    df = pd.DataFrame(dados_coletados)

    # 6. Inteligência de Dados: Validação e Duplicidade
    if not df.empty:
        # Cruza [CPF + Animal] para identificar repetidos
        duplicados = df.duplicated(subset=['CPF', 'Nome do Animal'], keep=False)
        
        status_registros = []
        for idx, row in df.iterrows():
            status = []
            # Evita marcar 'CPF não identificado' como duplicado genérico com outros não identificados
            if duplicados[idx] and row['CPF'] not in ['CPF não identificado', 'Não identificado']:
                status.append('DUPLICADO')
            
            if row.get('CPF') == 'CPF Inválido':
                status.append('CPF MATEMATICAMENTE INVÁLIDO')

            if not row.get('Assinatura Presente', True):
                status.append('PENDENTE ASSINATURA')
            
            if not status:
                status.append('OK')
                
            status_registros.append(' / '.join(status))
            
        df['Status_Registro'] = status_registros
        df.drop(columns=['Status_CPF'], inplace=True, errors='ignore')

        # Reordena as colunas conforme a especificação solicitada
        colunas_finais = ['Página', 'Nome Completo', 'CPF', 'Nome do Animal', 'Espécie', 'Assinatura Presente', 'Status_Registro']
        df = df[[col for col in colunas_finais if col in df.columns]]

    # 7. Exportação para Excel
    try:
        df.to_excel(arquivo_saida, index=False, engine='openpyxl')
        print(
            f"\nProcesso concluído com sucesso! Resultados salvos em '{arquivo_saida}'.")
    except Exception as e:
        print(f"\nErro ao salvar o arquivo Excel: {e}")
        print("Tentando salvar como CSV...")
        df.to_csv(arquivo_saida.replace('.xlsx', '.csv'), index=False)
        print(
            f"Resultados salvos em '{arquivo_saida.replace('.xlsx', '.csv')}'.")


def iniciar_gui():
    def selecionar_pdf():
        caminho = filedialog.askopenfilename(
            title="Selecione o arquivo PDF",
            filetypes=[("Arquivos PDF", "*.pdf")]
        )
        if caminho:
            entrada_pdf.delete(0, tk.END)
            entrada_pdf.insert(0, caminho)

    def executar_processamento():
        caminho_pdf = entrada_pdf.get()
        if not caminho_pdf or not os.path.exists(caminho_pdf):
            messagebox.showerror("Erro", "Por favor, selecione um arquivo PDF válido.")
            return

        botao_processar.config(state=tk.DISABLED)
        label_status.config(text="Status: Processando... Isso pode demorar alguns minutos.", fg="blue")

        # Rodar o processamento em uma thread separada evita que a interface congele
        def thread_processamento():
            try:
                # Salva o arquivo de resultado na mesma pasta onde está o PDF selecionado
                pasta_origem = os.path.dirname(caminho_pdf)
                arquivo_saida = os.path.join(pasta_origem, "resultado_formularios.xlsx")
                
                processar_pdf_formularios(caminho_pdf, arquivo_saida)
                
                root.after(0, lambda: label_status.config(text="Status: Concluído!", fg="green"))
                root.after(0, lambda: messagebox.showinfo("Sucesso", f"Processamento concluído!\nArquivo salvo em:\n{arquivo_saida}"))
            except Exception as e:
                root.after(0, lambda: label_status.config(text="Status: Erro durante o processamento!", fg="red"))
                root.after(0, lambda: messagebox.showerror("Erro", f"Ocorreu um erro:\n{e}"))
            finally:
                root.after(0, lambda: botao_processar.config(state=tk.NORMAL))

        threading.Thread(target=thread_processamento, daemon=True).start()

    root = tk.Tk()
    root.title("Processador de Formulários - Animais")
    root.geometry("800x600")
    root.resizable(True, True)
    
    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(frame, text="Arquivo PDF para Processar:").pack(anchor=tk.W)
    
    frame_arquivo = tk.Frame(frame)
    frame_arquivo.pack(fill=tk.X, pady=(5, 15))
    
    entrada_pdf = tk.Entry(frame_arquivo)
    entrada_pdf.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
    
    botao_procurar = tk.Button(frame_arquivo, text="Procurar...", command=selecionar_pdf)
    botao_procurar.pack(side=tk.RIGHT)

    botao_processar = tk.Button(frame, text="Iniciar Processamento", command=executar_processamento, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
    botao_processar.pack(fill=tk.X, pady=(10, 10))

    label_status = tk.Label(frame, text="Status: Aguardando seleção.", fg="gray")
    label_status.pack(anchor=tk.W)
    
    # --- CONSOLE DE LOGS INTERNO ---
    tk.Label(frame, text="Console de Execução e Auditoria:").pack(anchor=tk.W, pady=(15, 5))
    
    frame_console = tk.Frame(frame)
    frame_console.pack(fill=tk.BOTH, expand=True)
    
    scrollbar_console = tk.Scrollbar(frame_console)
    scrollbar_console.pack(side=tk.RIGHT, fill=tk.Y)
    
    texto_console = tk.Text(frame_console, wrap=tk.WORD, yscrollcommand=scrollbar_console.set, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 10))
    texto_console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar_console.config(command=texto_console.yview)

    # Ativa o redirecionamento global do Terminal para a Tela
    sys.stdout = RedirecionadorTerminal(texto_console)

    root.mainloop()

# --- EXECUÇÃO DO SCRIPT ---
if __name__ == "__main__":
    iniciar_gui()
