# Castra-DF - Processador de Formulários Mestre

Um sistema automatizado com interface gráfica moderna para processamento em lote de formulários em PDF. Ele utiliza Visão Computacional e OCR (Optical Character Recognition) para extrair dados de tutores e animais, validá-los e alimentar planilhas de banco de dados automaticamente.

## 🚀 Funcionalidades

- **Processamento em Lote:** Selecione um arquivo PDF individual ou uma pasta inteira.
- **Extração Inteligente (OCR):** Identifica e extrai Nome do Tutor, CPF, Nome do Animal e Espécie.
- **Validação de Dados:** Validação matemática de CPF e tratamento avançado contra falsos positivos.
- **Detecção de Assinatura:** Verifica automaticamente se o documento foi assinado.
- **Retomada Inteligente (Resume):** O sistema escaneia as planilhas existentes e pula automaticamente páginas já processadas anteriormente, poupando tempo em caso de interrupções.
- **Gestão Eficiente de Memória:** O progresso é salvo no disco a cada 10 páginas e a memória RAM é limpa, evitando travamentos em arquivos PDF com milhares de páginas.
- **Recuperação de Falhas (2ª Chance):** Aplica filtros de processamento de imagem (inversão de cores, correção de inclinação, CLAHE) caso a primeira leitura falhe.
- **Exportação para Excel:** Salva os registros em Bancos Mestres separados automaticamente por tipo (Cirúrgico, Antiparasitário, Anestésico) e gera um Relatório de Erros para revisão humana.
- **Painel de Revisão Interativo:** Cards dinâmicos e clicáveis na aba de erros para filtrar pendências visualmente. Correções feitas aqui atualizam o banco de dados mestre automaticamente em segundo plano.
- **Auto-Save e Limpeza Automática:** Padronização automatizada de textos/espécies ao carregar as planilhas e salvamento automático do trabalho a cada 60 segundos.
- **Gerenciamento de Duplicidades:** Interface gráfica avançada que permite buscar, editar células diretamente via duplo-clique e remover registros duplicados com um botão.
- **Interface Gráfica (GUI):** Desenvolvida com CustomTkinter, oferecendo suporte a temas Dark/Light, barra de progresso e abas com tabelas de visualização integradas.

## 📋 Pré-requisitos

Antes de executar o projeto, você precisará ter instalado:

1. **Python 3.8+**
2. **Tesseract-OCR:** É necessário instalar o executável do Tesseract no seu sistema operacional.
   - **Windows:** Baixe e instale via [UB-Mannheim Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki).
   - O código possui configuração dinâmica de caminho, detectando automaticamente se está rodando como um script ou como um `.exe` gerado pelo PyInstaller (neste caso, busca o Tesseract nativamente em uma pasta local `Tesseract-OCR`).

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
   - `BANCO_MESTRE_CIRURGICO.xlsx`
   - `BANCO_MESTRE_ANTIPARASITARIO.xlsx`
   - `BANCO_MESTRE_ANESTESICO.xlsx`
   - `REVISAO_MANUAL_CASTRA.xlsx`: Registros que necessitam de intervenção/revisão humana.
5. Use as abas **Base de Dados Acumulada** e **Relatório de Erros** para consultar os resultados diretamente pelo aplicativo.
6. Utilize a barra de pesquisa na aba de Base de Dados para filtrar rapidamente por Nome ou CPF, e edite informações incorretas diretamente na tabela (basta dar duplo-clique na célula).
7. Na aba de Erros, clique nos **Cards Coloridos** no topo da tela para filtrar rapidamente os problemas (ex: exibir apenas registros com CPFs inválidos ou sem assinatura).