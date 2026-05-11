# Castra-DF - Processador de Formulários Mestre

Um sistema automatizado com interface gráfica moderna para processamento em lote de formulários em PDF. Ele utiliza Visão Computacional e OCR (Optical Character Recognition) para extrair dados de tutores e animais, validá-los e alimentar planilhas de banco de dados automaticamente.

## 🚀 Funcionalidades

- **Processamento em Lote:** Selecione um arquivo PDF individual ou uma pasta inteira.
- **Extração Inteligente (OCR):** Identifica e extrai Nome do Tutor, CPF, Nome do Animal e Espécie.
- **Validação de Dados:** Validação matemática de CPF e tratamento avançado contra falsos positivos.
- **Detecção de Assinatura:** Verifica automaticamente se o documento foi assinado.
- **Recuperação de Falhas (2ª Chance):** Aplica filtros de processamento de imagem (inversão de cores, correção de inclinação, CLAHE) caso a primeira leitura falhe.
- **Exportação para Excel:** Salva os registros com sucesso em um Banco Mestre e gera um Relatório de Erros para revisão humana.
- **Interface Gráfica (GUI):** Desenvolvida com CustomTkinter, oferecendo suporte a temas Dark/Light, barra de progresso e abas com tabelas de visualização integradas.

## 📋 Pré-requisitos

Antes de executar o projeto, você precisará ter instalado:

1. **Python 3.8+**
2. **Tesseract-OCR:** É necessário instalar o executável do Tesseract no seu sistema operacional.
   - **Windows:** Baixe e instale via [UB-Mannheim Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki).
   - O caminho padrão configurado no código é `C:\Users\07049770108\AppData\Local\Programs\Tesseract-OCR\tesseract.exe`. Ajuste no arquivo `processador_formularios.py` caso esteja em um local diferente.

## 🛠️ Instalação

1. Clone o repositório ou baixe os arquivos fonte.
2. Recomenda-se a criação de um ambiente virtual:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # No Windows
   ```
3. Instale as bibliotecas necessárias executando:
   ```bash
   pip install opencv-python numpy pandas pytesseract Pillow pypdfium2 customtkinter openpyxl
   ```

## 💻 Como Usar

Execute o arquivo principal para abrir a interface gráfica:

```bash
python processador_formularios.py
```

1. Na aba **"Processar PDFs"**, clique em "Procurar Arquivo" ou "Procurar Pasta".
2. Clique em **"INICIAR"**.
3. Acompanhe a barra de progresso e o log no console.
4. Ao final, os dados estarão salvos no mesmo diretório do arquivo de origem:
   - `BANCO_MESTRE_CASTRA.xlsx`: Registros processados com sucesso.
   - `REVISAO_MANUAL_CASTRA.xlsx`: Registros que necessitam de intervenção/revisão humana.
5. Use as abas **Base de Dados Acumulada** e **Relatório de Erros** para consultar os resultados diretamente pelo aplicativo.