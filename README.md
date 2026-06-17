# Pipeline de Produção Energética Marítima

Projeto acadêmico de processamento, transformação e análise de dados de produção energética marítima brasileira (2016-2026) com integração MySQL e visualizações gráficas.

## 📋 Sobre o Projeto

Este projeto implementa um pipeline completo de **Ciência de Dados** que:
- Processa multiple arquivos CSV com dados de produção de óleo, gás e água em plataformas marítimas
- Normaliza e valida dados de diversas fontes (ANP - Agência Nacional do Petróleo)
- Armazena dados em banco de dados MySQL com validação de integridade
- Gera estatísticas e visualizações sobre a produção energética marítima brasileira

## 🏗️ Estrutura do Projeto

```
DadosProducaoEnergetica/
├── processar_dados.py              # Script de ETL (extract, transform, load)
├── pipeline_mysql.py               # Pipeline MySQL com análises
├── producao_maritima_tratada.csv   # Dados consolidados e tratados
├── dados/                          # Dados brutos por ano
│   ├── producao-mar-2016-2018.csv
│   ├── producao-mar-2019.csv
│   ├── producao-mar-2020.csv
│   ├── producao-mar-2021.csv
│   ├── producao-mar-2022.csv
│   ├── producao-mar-2023.csv
│   ├── producao-mar-2025 (1).csv
│   ├── producao-mar-2026.csv
│   └── producao_por_poco_2024.csv
└── resultados_mysql/               # Outputs (gráficos, logs)
```

## 📊 Dados

### Fonte
Dados públicos da **Agência Nacional do Petróleo, Gás Natural e Biocombustíveis (ANP)**

### Período
2016 a 2026 (com foco em produção marítima)

### Colunas Principais
- **Identificação**: ano, mes_ano, competencia
- **Localização**: estado, bacia, campo, poço, ambiente, instalação
- **Produção**: óleo, condensado, gás associado/não-associado, água
- **Injeção**: gás, água para recuperação secundária, água para descarte, CO₂, nitrogênio, vapor, polímeros, outros fluidos

### Características de Tratamento
- ✅ Suporte a múltiplas codificações (UTF-8, UTF-8-sig, ISO-8859-1)
- ✅ Normalização de nomes de colunas (snake_case, sem acentuação)
- ✅ Conversão robusta de números brasileiros (vírgula decimal, pontos de milhar)
- ✅ Remoção de duplicatas
- ✅ Validação de dados obrigatórios

## 🚀 Quickstart

### Requisitos
- Python 3.8+
- MySQL Server 5.7+
- pip

### Instalação

1. **Clone o repositório:**
```bash
git clone <seu-repositorio>
cd DadosProducaoEnergetica
```

2. **Crie um ambiente virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate  # Windows
```

3. **Instale as dependências:**
```bash
pip install pandas mysql-connector-python matplotlib numpy
```

### Configuração do Banco de Dados

Edite `pipeline_mysql.py` e configure as credenciais MySQL:

```python
DB_CONFIG = {
    "host": "localhost",
    "user": "seu_usuario",
    "password": "sua_senha",
    "database": "producao_energetica",  # será criado automaticamente
}
```

### Uso

**Etapa 1: Processar e unificar dados**
```bash
python processar_dados.py
```
Saída: `producao_maritima_tratada.csv`

**Etapa 2: Carregar no MySQL e gerar análises**
```bash
python pipeline_mysql.py
```

Saída:
- Banco MySQL com tabela `producao_maritima`
- `resultados_mysql/pipeline.log` - Log de execução
- `resultados_mysql/grafico_pizza.png` - Distribuição por ano
- `resultados_mysql/grafico_area.png` - Evolução por estado

## 📈 Funcionalidades Principais

### processador_dados.py
- **Carregamento multi-encoding**: Detecta automaticamente a codificação dos CSVs
- **Normalização de colunas**: Converte nomes para padrão snake_case
- **Parsing de números brasileiros**: `1.234,56` → `1234.56`
- **Consolidação**: Combina múltiplos anos em um único dataset
- **Limpeza**: Remove duplicatas e valida dados obrigatórios

### pipeline_mysql.py
- **Conexão MySQL**: Criação automática de estrutura
- **Validação de integridade**: Chave primária baseada em (competencia, estado, bacia, campo, poço, fonte)
- **Upsert de dados**: Atualiza registros existentes
- **Estatísticas**: Média, mediana, desvio padrão da produção
- **Visualizações**:
  - Gráfico de pizza: distribuição de produção por ano
  - Gráfico de área: evolução temporal por estado

## 📊 Visualizações Geradas

### Gráfico de Pizza (grafico_pizza.png)
Mostra a proporção relativa de produção de óleo por ano (2016-2026)

### Gráfico de Área Empilhada (grafico_area.png)
Evolução da produção de óleo ao longo dos anos, separada por estado produtor

## 🔒 Configuração de Segurança

⚠️ **IMPORTANTE**: As credenciais MySQL estão hardcoded em `pipeline_mysql.py`. Para produção:

```python
# Use variáveis de ambiente
import os
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": "producao_energetica",
}
```

## 📝 Formato do CSV Tratado

| Campo | Tipo | Descrição |
|-------|------|-----------|
| ano | INT | Ano da produção |
| mes_ano | CHAR(7) | Mês/ano (MM/YYYY) |
| competencia | DATE | Data no formato YYYY-MM-DD |
| estado | VARCHAR(100) | UF produtor (BA, CE, AL, etc) |
| bacia | VARCHAR(100) | Bacia sedimentar |
| campo | VARCHAR(120) | Campo petrolífero |
| poço | VARCHAR(120) | Identificador do poço |
| ambiente | VARCHAR(40) | Ambiente (Mar, Terra) |
| instalacao | VARCHAR(180) | Plataforma/estação |
| producao_oleo_m3 | DECIMAL(18,5) | Produção de óleo (m³) |
| producao_condensado_m3 | DECIMAL(18,5) | Produção de condensado (m³) |
| injecao_gas_mm3 | DECIMAL(18,5) | Injeção de gás (mm³) |
| ... | DECIMAL(18,5) | Outras métricas de produção/injeção |

## 📚 Dependências

- **pandas**: Processamento de dados
- **mysql-connector-python**: Conexão com MySQL
- **matplotlib**: Visualizações gráficas
- **numpy**: Operações numéricas

## 📄 Licença

Projeto acadêmico - Quarto ano, Ciência de Dados

## 📞 Suporte

Para dúvidas ou problemas, verifique:
1. Se MySQL está rodando
2. Se as credenciais estão corretas
3. Se todos os CSVs estão na pasta `dados/`
4. O arquivo `resultados_mysql/pipeline.log` para detalhes de erros
