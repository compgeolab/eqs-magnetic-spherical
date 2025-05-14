#!/bin/bash

# ------------------------------------------------------------------------------
# Script de limpeza e pré-processamento de dados da ANM
# ------------------------------------------------------------------------------

# Defina os arquivos de entrada e saída
ARQUIVO_ENTRADA="magnetometria-bacia-do-parana.csv"
ARQUIVO_SAIDA="dados_tratados.csv"

# ------------------------------------------------------------------------------
# 1. Contar o número de linhas do arquivo de entrada
# ------------------------------------------------------------------------------

echo "Contando o número de linhas no arquivo de entrada:"
wc -l "$ARQUIVO_ENTRADA"
echo "--------------------------------------------------"

# ------------------------------------------------------------------------------
# 2. Extrair colunas 6, 7 e 10 a partir da linha 50
# ------------------------------------------------------------------------------

echo "Extraindo colunas 6, 7 e 10 a partir da linha 50..."
awk -F, 'NR > 49 { print $6","$7","$10 }' "$ARQUIVO_ENTRADA" > "$SAIDA_COLUNAS_6_7_10"
echo "Salvo em: $SAIDA_COLUNAS_6_7_10"

# ------------------------------------------------------------------------------
# 3. Extrair colunas 6, 7 e 18 a partir da linha 50
# ------------------------------------------------------------------------------

echo "Extraindo colunas 6, 7 e 18 a partir da linha 50..."
awk -F, 'NR > 49 { print $6","$7","$18 }' "$ARQUIVO_ENTRADA" > "$SAIDA_COLUNAS_6_7_18"
echo "Salvo em: $SAIDA_COLUNAS_6_7_18"
echo "--------------------------------------------------"

# ------------------------------------------------------------------------------
# 4. Verificar presença da string 'PR-FAS'
# ------------------------------------------------------------------------------

echo "Verificando presença da string 'PR-FAS'..."
grep "PR-FAS" "$ARQUIVO_ENTRADA"
echo "--------------------------------------------------"

# ------------------------------------------------------------------------------
# 5. Remover ocorrências da string 'PR-FAS'
# ------------------------------------------------------------------------------

echo "Removendo ocorrências da string 'PR-FAS'..."
sed -i 's/.*,PR-FAS//' "$ARQUIVO_ENTRADA"
echo "Limpeza concluída."
echo "--------------------------------------------------"

echo "Script finalizado com sucesso!"
