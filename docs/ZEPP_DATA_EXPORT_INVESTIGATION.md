# Investigação: Exportação de Dados Zepp Cloud - Dados Históricos Antes de Junho 2024

## Problema Identificado

A exportação de dados do Zepp Cloud realizada no projeto contém apenas dados até **junho de 2024**, mesmo que o participante tenha histórico de dados muito mais anterior (desde 2021-2022).

**Arquivo afetado**: `data/raw/P000001/zepp/3088235680_1762500387835.zip`

- Contém apenas registros de SLEEP, ACTIVITY, HEARTRATE até ~junho 2024
- Data snapshot de extração: 2025-11-07 (muito recente)
- Intervalo temporal dos dados: até junho 2024

## Questões Investigadas

### 1. Há limite de retenção de dados no Zepp Cloud?

Segundo políticas padrão de nuvem:

- **Zepp Cloud pode ter uma política de retenção de dados de ~1 ano**
- Dados mais antigos que junho 2024 podem ter sido automaticamente deletados da conta cloud
- Sem documentação oficial pública, isso permanece uma incógnita

### 2. Caminhos Alternativos para Obter Dados Históricos

#### A. Contato Direto com Suporte Zepp

- **Opção**: Solicitar manualmente ao suporte Zepp dados históricos (GDPR/Privacy Request)
- **Processo típico**:
  1. Acessar conta Zepp
  2. Enviar "Data Subject Access Request" (DSAR) ao suporte
  3. Solicitar especificamente dados anteriores a junho 2024
  4. Esperar 30-45 dias para resposta
- **Probabilidade**: BAIXA a MÉDIA - dados podem estar permanentemente deletados

#### B. Verificar Backup Local no Relógio/Watch

- **Opção**: Extrair dados diretamente do dispositivo Zepp (relógio/band)
- **Requer**:
  - Acesso físico ao dispositivo
  - Ferramentas de backup local (se disponíveis no SO do watch)
- **Desafio**: Muitos watches Zepp não têm modo de exportação local

#### C. Verificar Histórico em Dispositivos iOS/Android

- **Opção**: Backup do aplicativo móvel Zepp
- **Processo**:
  1. Verificar backup iCloud (iOS) ou Google Drive (Android)
  2. Extrair dados da base de dados local do app
  3. Requerer senha de backup se encriptado
- **Probabilidade**: MÉDIA - dados locais podem estar ali

#### D. API Zepp (Se Disponível)

- **Status**: Zepp Health (empresa chinesa) não publica API pública clara
- **Alternativa**: Alguns projetos open-source tentam reverse-engineer a API
  - Exemplo: `https://github.com/` (procurar por "Zepp API reverse")
- **Risco**: Violação de ToS, dados podem estar fora do escopo da API

#### E. Repositório de Código Open Source

- **Pesquisa**: Verificar repositórios como:
  - Projetos Zepp data export em GitHub
  - Ferramentas de migração de relógios Zepp/Xiaomi
  - Scripts de backup não-oficial
- **Exemplo search**: `site:github.com zepp data export historical api`

### 3. Análise de Políticas de Retenção - Indústria

**Comparação com competidores**:

- **Garmin**: Retém dados indefinidamente (sincronizados)
- **Fitbit**: Retém dados por ~indefinido se ativo
- **Apple Health**: Retém dados indefinidamente no iCloud
- **Huami/Zepp**: **Documentação não pública** - suspeita-se ~12 meses

## Recomendações Práticas

### Curto Prazo (Agora)

1. **Verificar arquivo JSON no ZIP Zepp**

   ```bash
   unzip -l data/raw/P000001/zepp/3088235680_1762500387835.zip | grep -i json
   ```

   - Pode haver metadados com informações de data de exportação/sincronização

2. **Contatar Suporte Zepp**

   - Email: support@zepp.com ou support@zepp.cn
   - Solicitar: "Historical data export - data before 2024-06-01"
   - Mencionar: Pesquisa/ADHD study com IRB approval (se houver)

3. **Verificar conta Zepp online**
   - Login em https://app.zepp.com
   - Navegar para seção de "Dados" ou "Histórico"
   - Verificar se há visualização de dados anteriores a junho 2024

### Médio Prazo (1-2 semanas)

1. **Extrair dados do backup local do smartphone**

   - iOS: Buscar Database SQLite em backup do iTunes
   - Android: ADB para extrair dados locais do app Zepp

2. **Pesquisar ferramentas comunitárias**
   - GitHub search: `zepp data export`, `zepp backup`, `huami data`
   - Avaliar viabilidade e riscos

### Longo Prazo (Planejamento Futuro)

1. **Documentar estratégia de coleta**

   - Para estudos futuros, solicitar exportação trimestral (não esperar acumular)
   - Manter backups locais regulares

2. **Alternativas de Wearables**
   - Se replicar estudo: considerar Garmin (melhor retenção de dados)
   - Apple Watch com HealthKit (dados persistentes)

## Status: Inconcluso

Sem acesso à documentação oficial Zepp ou resposta de suporte, não é possível confirmar:

- ✗ Se dados foram deletados automaticamente
- ✗ Se há limite de retenção configurável
- ✗ Se há opção de exportação de dados arquivados

## Arquivos Relevantes

- ZIP de exportação: `data/raw/P000001/zepp/3088235680_1762500387835.zip` (20 arquivos)
- Intervalo detectado: ~2024-06-01 até presente
- Dados faltando: 2021/2022 - 2024-06-01

---

**Última atualização**: 2025-11-07
**Investigador**: Sistema de análise de ETL
