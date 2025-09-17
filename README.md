# Desafio de Seleção — Desenvolvedor Estagiário **Full-Stack + IA** - FASA/UNICAP

Construa uma aplicação **full-stack** simples, composta por:
- **Front-end** em **React _ou_ Angular**, que consome uma API (Back-End) e exibe os resultados.
- **Back-end** em **Python + FastAPI**, que **processa dados localmente** (modelo/algoritmo na própria API) **ou** via **API externa** (ex.: Hugging Face Inference API, OpenAI, Google Vision, etc. — plano gratuito).
> Obs.: No caso de uso de API externa, o desenvolvimento do Back-end, ainda que apenas para processamento de dados da API externa é obrigatória!

O desafio é **genérico**, mas todos os pontos abaixo são obrigatórios e detalhados para avaliarmos arquitetura, código e clareza.

---

## 1) Objetivo
Entregar um **MVP funcional** que:
1. Receba um input do usuário (_texto_, _imagem_ ou _áudio_).
2. Envie esse input ao **back-end (FastAPI)** por HTTP.
3. O back-end **processe**:
   - **Localmente** (ex.: análise de sentimento com um modelo simples; NER com spaCy; OCR local; classificação com scikit-learn; ONNX Runtime; etc.), **ou**
   - **Externamente** (chamando uma **API de IA**, igualmente realizando tarefas com texto, imagem, áudio, etc.).
4. Retorne um **resultado estruturado** ao front-end (JSON).
5. O **front-end** exiba o resultado (processar o JSON obtido para interagir com a UI) de forma clara (inclua estado de **loading**, tratamento de **erros** e UI mínima).

> Observação: escolha **uma tarefa de IA** (ex.: **sentiment analysis**, **classificação de texto**, **extração de entidades**, **resumo**, **OCR**, **caption de imagem**, etc.). Valorize a **clareza do fluxo**.

---

## 2) Requisitos Técnicos (Obrigatórios)

### Back-end (Python + FastAPI)
- **Python 3.10+** e **FastAPI**.
- Endpoints mínimos:
  - `POST /api/v1/analyze`  
    - Aceitar **JSON** (para texto) **ou** `multipart/form-data` (para áudio/imagem).  
    - Corpo **sugerido** (JSON):
      ```json
      {
        "task": "sentiment|ner|ocr|caption|custom",
        "input_text": "string opcional",
        "use_external": true,
        "options": { "lang": "pt" }
      }
      ```
    - Resposta **exemplo**:
      ```json
      {
        "id": "uuid",
        "task": "sentiment",
        "engine": "external:hf-distilbert-sst2" ,
        "result": { "label": "POSITIVE", "score": 0.98 },
        "elapsed_ms": 123,
        "received_at": "2025-09-15T12:34:56Z"
      }
      ```
  - `GET /api/v1/healthz` → `{"status":"ok"}`
- **CORS** habilitado para o front.
- **Tratamento de erros**: retornar mensagens claras e `HTTP status` adequados.
- **Logs** básicos (requisição, erro, duração).

### Integração de IA (Escolha 1 caminho ou implemente ambos)
- **Local** (exemplos):  
  - spaCy (NER), scikit-learn (sentiment clássico), transformers em modo local/CPU, ONNX Runtime, Tesseract para OCR etc.
- **Externa** (exemplos):  
  - Hugging Face Inference API, OpenAI, Google Vision, Cohere etc.  
  - **Nunca** exponha chaves no repositório; use variáveis de ambiente.

### Front-end (React **ou** Angular)
- Uma página com:
  - Campo de texto **ou** upload de arquivo (se a tarefa exigir imagem/áudio/pdf).
  - Botão **Analisar**.
  - Indicação de **carregando** (spinner/skeleton, isso enquanto aguarda resposta da requisição).
  - Exibição do **resultado amigável** (ex.: label/score, entidades destacadas, objetos segmentados, texto extraído, etc.).
  - Exibição de **erros** amigáveis (ex.: rate limit, validação).
- Organização mínima:
  - **React**: Vite/CRA/Next (página única/SPA é suficiente), fetch/axios, componentes simples.
  - **Angular**: CLI, serviço para HTTP, módulo e componente(s) simples.

---

## 3) Checklist avaliativo
- [ ] `POST /api/v1/analyze` (rota para requisição) funcional (texto **ou** arquivo).
- [ ] Processamento **local** **ou** **externo** realmente executado (não simular/mockar).
- [ ] Front chama a API e **exibe o resultado** de forma clara.
- [ ] Estados de **loading** e **erro** implementados no front.
- [ ] Video de no máximo 5min demonstrando a aplicação sendo executada e explicação do código (entregar por e-mail).
- [ ] **README** com passos de setup/execução e descrição da tarefa de IA escolhida.
- [ ] Código versionado no Git com histórico de commits compreensível.
- [ ] Tanto o código do front-end quanto back-end devem estar modularizado em camadas e seguindo SOLID.

---

## 4) O que Entregar
1. **Repositório Git** público (ou link de acesso) contendo:
   - `/backend` (projeto back-end na pasta raiz ``backend`` e escrito com Python/FastAPI)  
   - `/frontend` (rojeto front-end na pasta raiz ``frontend``React **ou** Angular)
   - `README.md` na raiz explicando:
     - Visão geral (arquitetura e decisão Local vs Externa).
     - Como rodar **backend** e **frontend** 
     - Exemplos de requisição e resposta (curl/HTTPie/Postman).

---

## 5) Avaliação (pesos)
- **Qualidade do código & organização** 
  Estrutura limpa, tipagem/annotations, docstrings, padrões de projeto simples.
- **Arquitetura & boas práticas** 
  Camadas separadas (rota → serviço → cliente IA), env vars, CORS, erros, logs.
- **API design & contratos** 
  Claros, consistentes, status codes corretos, OpenAPI coerente.
- **Front-end & UX mínima** 
  Estados, mensagens, estrutura de componentes/serviços.
- **Integração de IA**
  Execução real (local **ou** externa), mapeamento do resultado no contrato.

---

## 6) Regras & Restrições
- Se usar **API externa**, respeite limites do **free tier** e trate erros de quota.
- O **processamento deve ocorrer de fato** (não valem dados mockados).
- Mantenha o projeto **rodando localmente** (cloud não é um requisito obrigatório).
- O vídeo mencionado anteriormente deve ser enviado por e-mail.

---

## 7) Bônus (Diferenciais)
- **Ambos os modos**: local **e** externo (com _feature flag_ `USE_EXTERNAL=true/false`).
- **Cache** simples no back-end (ex.: in-memory) para inputs repetidos.
- **Rate limiting** básico.
- **Histórico** de análises em memória/Banco (exibir no front).
- **CI** (GitHub Actions) para lint/test.
- **Teste E2E** simples (ex.: Playwright/Cypress para o front).
- **Deploy opcional** (Railway/Render/Fly.io/EC2) com instruções (não obrigatório).

---

## 8) Sugestões de Tarefas de IA (escolha 1)
- **Texto**
  - Análise de sentimento (pt/en)
  - Extração de entidades (NER)
  - Resumo de textos
  - Agentes e RAG
- **Imagem**
  - OCR de imagem (PNG/JPG) → retorna texto
  - Geração de legenda (image caption)
- **Custom**: qualquer tarefa simples ou multimodal (que envolva mais de um tipo de dado), desde que demonstre o fluxo ponta-a-ponta.

---

## 9) Prazos & Entrega
- **Tempo sugerido**: 20/09/2025.
- Envie **link do repositório** e um **vídeo curto (5 min)** mostrando o uso e explicação de todo o código.
