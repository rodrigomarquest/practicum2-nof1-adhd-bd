# Zepp Data Recovery - Checklist de Ações

## 🔍 Situação Atual

- **Data do snapshot ETL**: 2025-11-07
- **Dados presentes no ZIP**: Até ~2024-06-01
- **Dados faltando**: 2021-2022 até 2024-06-01 (~2 anos)
- **Questão**: Dados deletados ou não sincronizados?

---

## ✅ Ações Imediatas (Esta Semana)

### 1. Verificar Metadados do ZIP Zepp

```bash
# Listar conteúdo do ZIP
unzip -l data/raw/P000001/zepp/3088235680_1762500387835.zip

# Procurar por arquivo de manifesto/metadata
unzip -l data/raw/P000001/zepp/3088235680_1762500387835.zip | grep -E "manifest|meta|index|readme"

# Extrair e examinar datas dos registros
unzip -p data/raw/P000001/zepp/3088235680_1762500387835.zip | grep -o "[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}" | sort -u | head -5
```

### 2. Verificar Conta Zepp Online

1. Acessar https://app.zepp.com
2. Login com credenciais do participante
3. Navegar: **Dados/Histórico/Estatísticas**
4. Verificar:
   - Data mais antiga disponível na visualização
   - Se há opções de filtro de data histórica
   - Se aparece aviso de "dados arquivados"

### 3. Contato com Suporte Zepp (Email)

**Destinatário**: support@zepp.com ou support@zepp.cn  
**Assunto**: "Request for Historical Data Export - Research Study"

**Corpo do Email**:

```
Dear Zepp Support Team,

I am requesting historical wearable data for research purposes.

Details:
- Account: [email da conta]
- Device: [tipo de relógio/band]
- Requested date range: 2021-01-01 to 2024-12-31
- Current export contains only data until ~2024-06-01

Questions:
1. Is there a data retention policy (e.g., 12 months)?
2. Can historical data before 2024-06-01 be recovered/exported?
3. If data was deleted, is there an archive recovery option?

This is for an IRB-approved ADHD research study. Any assistance
would be appreciated.

Best regards,
[Nome]
```

---

## 📱 Alternativa: Recuperação de Backup Local

### iOS (iPhone)

1. Conectar iPhone ao iTunes/Finder (macOS)
2. Fazer backup completo (não iCloud)
3. Usar ferramenta: **iPhone Backup Extractor**
   - Procurar por database do Zepp app:
     ```
     Library/Application Support/zepp/data.db
     ```
4. Extrair SQLite database
5. Verificar tabelas de histórico de dados

**Tools**:

- iBackup Extractor (gratuito)
- PhoneRescue
- SQLite Studio (para ler banco de dados)

### Android

1. Ativar **USB Debugging** no telefone
2. Usar ADB (Android Debug Bridge):
   ```bash
   adb shell "pm dump com.huami.watch" | grep -i data
   adb pull /data/data/com.huami.watch/databases/
   ```
3. Extrair e examinar SQLite databases
4. Procurar por tabelas de HR, Sleep, Activity

**Tools**:

- Android Studio (ADB incluído)
- SQLite Browser

---

## 🔗 Ferramentas Open Source (Pesquisar)

### GitHub Search Queries

1. `zepp data export`
2. `zepp backup tools`
3. `huami watch export`
4. `xiaomi watch data recovery`
5. `zepp cloud api reverse engineer`

### Projetos Potenciais

- Procurar por repos com ⭐ 50+ estrelas
- Ler issues/discussions sobre data retention
- Verificar código para API endpoints descobertos

---

## ⚠️ Cuidados Importantes

### GDPR/Privacy Compliance

- Solicitar dados via "Data Subject Access Request" (DSAR)
- Zepp é obrigado por lei a fornecer em 30 dias
- Documentar o pedido para auditoria

### Termos de Serviço

- ✓ OK: Usar ferramentas oficiais Zepp
- ✓ OK: Contatar suporte
- ⚠️ CUIDADO: Reverse-engineering da API pode violar ToS
- ✗ PROIBIDO: Acessar conta de outro usuário

### Integridade de Dados

- Manter backup do ZIP original
- Documentar fonte de todos os dados importados
- Se misturar múltiplas fontes, deixar claro qual é qual

---

## 📊 Próximas Etapas se Dados Recuperados

1. **Criar novo snapshot para dados históricos**

   ```bash
   mkdir -p data/raw/P000001/zepp_archived/
   # Copiar dados recuperados aqui
   ```

2. **Atualizar documentação de ETL**

   - Anotar qual é fonte original vs. recuperada
   - Adicionar campo de "data_source" em features

3. **Re-executar pipeline**
   ```bash
   make extract PID=P000001 SNAPSHOT=2025-11-07-WITH-ARCHIVE
   make biomarkers PID=P000001 SNAPSHOT=2025-11-07-WITH-ARCHIVE
   ```

---

## 📞 Contatos Úteis

| Organização        | Contato           | Tipo          |
| ------------------ | ----------------- | ------------- |
| Zepp Support       | support@zepp.com  | Email         |
| Zepp Support CN    | support@zepp.cn   | Email (China) |
| Huami (Fabricante) | support@huami.com | Email         |
| Privacy Officer    | privacy@zepp.com  | GDPR Request  |

---

## 🎯 Timeline Recomendado

| Data         | Ação                                                   | Responsável |
| ------------ | ------------------------------------------------------ | ----------- |
| **Hoje**     | Executar checklist imediato (itens 1-3)                | Pesquisador |
| **Semana 1** | Receber resposta de suporte                            | Zepp        |
| **Semana 2** | Tentar recuperação local se needed                     | Pesquisador |
| **Semana 3** | Avaliar ferramentas open source                        | Dev         |
| **Semana 4** | Decisão final: prosseguir com dados atuais ou aguardar | Equipe      |

---

## 📝 Notas Adicionais

### Por que não há mais dados?

**Hipóteses** (em ordem de probabilidade):

1. **Zepp Cloud retém dados por ~12 meses** (padrão de indústria)
   - Provavelmente política não-documentada
2. **Conta foi resetada/re-sincronizada em 2024**
   - Pode ter perdido histórico anterior
3. **Watch foi repaired/replaced em 2024**

   - Novo device, novo histórico começa

4. **User mudou de watch Zepp em 2024**
   - Dados do watch antigo não migrados

### Impacto no Estudo

- ✗ Se dados forem irrecuperáveis: limitar análise a ~18 meses de dados
- ✓ Se dados forem recuperados: análise temporal completa possível

---

**Status**: 🔄 Aguardando investigação
**Prioridade**: MÉDIA (afeta completude dos dados, não inviabiliza estudo)
