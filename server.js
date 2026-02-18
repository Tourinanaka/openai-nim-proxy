// server.js - OpenAI to NVIDIA NIM API Proxy (診断モード)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

if (!NIM_API_KEY) {
  console.error('FATAL: NIM_API_KEY environment variable is not set');
  process.exit(1);
}

const SHOW_REASONING = true;
const ENABLE_THINKING_MODE = true;

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'z-ai/glm4.7',
  'claude-3-sonnet': 'z-ai/glm5',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

const verifiedModels = new Map();

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'OpenAI to NVIDIA NIM Proxy (DIAGNOSTIC MODE)' });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model, object: 'model', created: Math.floor(Date.now() / 1000), owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// ============================================================
// メイン：診断ログ付き
// ============================================================
app.post('/v1/chat/completions', async (req, res) => {
  const requestId = `req-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // ★ ログ1: JanitorAIから来た生リクエストをすべて表示
    console.log('\n' + '='.repeat(80));
    console.log(`[${requestId}] ★ INCOMING REQUEST FROM JANITORAI`);
    console.log('='.repeat(80));
    console.log(`  Model requested : ${model}`);
    console.log(`  Temperature     : ${temperature}`);
    console.log(`  Max tokens      : ${max_tokens}`);
    console.log(`  Stream          : ${stream}`);
    console.log(`  Messages count  : ${messages?.length}`);
    // メッセージの先頭と末尾だけ表示（長すぎるので）
    if (messages && messages.length > 0) {
      messages.forEach((msg, i) => {
        const preview = msg.content ? msg.content.substring(0, 150) : '(empty)';
        console.log(`  Message[${i}] role=${msg.role} | ${preview}${msg.content?.length > 150 ? '...' : ''}`);
      });
    }

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: { message: "'messages' is required and must be a non-empty array", type: 'invalid_request_error', code: 400 }
      });
    }

    // Model mapping
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      if (verifiedModels.has(model)) {
        const cached = verifiedModels.get(model);
        if (cached) nimModel = cached;
      } else {
        try {
          const probeResponse = await axios.post(`${NIM_API_BASE}/chat/completions`, {
            model: model, messages: [{ role: 'user', content: 'test' }], max_tokens: 1
          }, {
            headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
            timeout: 10000, validateStatus: (s) => s < 500
          });
          if (probeResponse.status >= 200 && probeResponse.status < 300) {
            nimModel = model; verifiedModels.set(model, model);
          } else { verifiedModels.set(model, null); }
        } catch (e) { verifiedModels.set(model, null); }
      }
      if (!nimModel) {
        const ml = model.toLowerCase();
        if (ml.includes('gpt-4') || ml.includes('claude-opus') || ml.includes('405b')) nimModel = 'meta/llama-3.1-405b-instruct';
        else if (ml.includes('claude') || ml.includes('gemini') || ml.includes('70b')) nimModel = 'meta/llama-3.1-70b-instruct';
        else nimModel = 'meta/llama-3.1-8b-instruct';
      }
    }

    // ★ ログ2: NIMに送るリクエストを完全表示
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.85,
      max_tokens: max_tokens || 1009024,
      stream: stream || false,
      ...(ENABLE_THINKING_MODE && { chat_template_kwargs: { thinking: true } })
    };

    console.log('\n' + '-'.repeat(80));
    console.log(`[${requestId}] ★ OUTGOING REQUEST TO NVIDIA NIM`);
    console.log('-'.repeat(80));
    console.log(`  NIM Model       : ${nimRequest.model}`);
    console.log(`  Temperature     : ${nimRequest.temperature}`);
    console.log(`  Max tokens      : ${nimRequest.max_tokens}`);
    console.log(`  Stream          : ${nimRequest.stream}`);
    console.log(`  Thinking mode   : ${nimRequest.chat_template_kwargs ? JSON.stringify(nimRequest.chat_template_kwargs) : 'NOT SET'}`);
    console.log(`  Full NIM body (excluding messages):`);
    const { messages: _m, ...nimWithoutMessages } = nimRequest;
    console.log(`  ${JSON.stringify(nimWithoutMessages, null, 2)}`);

    // ★ リクエスト送信
    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
        timeout: 120000,
        responseType: stream ? 'stream' : 'json'
      }
    );

    // ★ ログ3: NIMからの成功レスポンス
    console.log('\n' + '-'.repeat(80));
    console.log(`[${requestId}] ★ NIM RESPONSE - SUCCESS (status ${response.status})`);
    console.log('-'.repeat(80));

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;
      let chunkCount = 0;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();

        // ★ 最初の数チャンクだけ生データを表示
        chunkCount++;
        if (chunkCount <= 3) {
          console.log(`[${requestId}] Stream chunk #${chunkCount}: ${chunk.toString().substring(0, 300)}`);
        }

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;
          if (line.includes('[DONE]')) {
            if (SHOW_REASONING && reasoningStarted) {
              res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: '</think>\n\n' } }] })}\n\n`);
              reasoningStarted = false;
            }
            res.write('data: [DONE]\n\n');
            return;
          }
          try {
            const data = JSON.parse(line.slice(6));
            if (data.choices?.[0]?.delta) {
              const reasoning = data.choices[0].delta.reasoning_content;
              const content = data.choices[0].delta.content;
              if (SHOW_REASONING) {
                let combinedContent = '';
                if (reasoning && !reasoningStarted) { combinedContent = '<think>\n' + reasoning; reasoningStarted = true; }
                else if (reasoning) { combinedContent = reasoning; }
                if (content && reasoningStarted) { combinedContent += '</think>\n\n' + content; reasoningStarted = false; }
                else if (content) { combinedContent += content; }
                if (combinedContent) data.choices[0].delta.content = combinedContent;
              } else {
                data.choices[0].delta.content = content || '';
              }
              delete data.choices[0].delta.reasoning_content;
            }
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) { res.write(line + '\n'); }
        });
      });

      response.data.on('end', () => {
        console.log(`[${requestId}] Stream ended. Total chunks: ${chunkCount}`);
        if (SHOW_REASONING && reasoningStarted) {
          res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: '</think>\n\n' } }] })}\n\n`);
        }
        res.end();
      });

      response.data.on('error', (err) => {
        console.error(`[${requestId}] ★ STREAM ERROR: ${err.message}`);
        try { res.write(`data: ${JSON.stringify({ error: { message: err.message } })}\n\n`); } catch (_) {}
        res.end();
      });

    } else {
      // Non-streaming
      console.log(`[${requestId}] Response data:`, JSON.stringify(response.data, null, 2).substring(0, 1000));

      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          return { index: choice.index, message: { role: choice.message.role, content: fullContent }, finish_reason: choice.finish_reason };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }

  } catch (error) {
    // ★★★ ログ4: エラーの完全情報 ★★★
    console.error('\n' + '!'.repeat(80));
    console.error(`[${requestId}] ★★★ ERROR DETAILS ★★★`);
    console.error('!'.repeat(80));
    console.error(`  Error message   : ${error.message}`);
    console.error(`  Error code      : ${error.code}`);
    console.error(`  HTTP status     : ${error.response?.status}`);
    console.error(`  Status text     : ${error.response?.statusText}`);

    // ★ NIMが返したエラーボディ（これが一番重要）
    if (error.response?.data) {
      console.error(`  ★ NIM ERROR BODY:`);
      if (typeof error.response.data === 'string') {
        console.error(`    ${error.response.data}`);
      } else if (error.response.data.pipe) {
        // ストリームの場合は読み取る
        let errorBody = '';
        try {
          for await (const chunk of error.response.data) {
            errorBody += chunk.toString();
          }
          console.error(`    (stream error body): ${errorBody}`);
        } catch (e2) {
          console.error(`    (could not read stream error body)`);
        }
      } else {
        console.error(`    ${JSON.stringify(error.response.data, null, 2)}`);
      }
    }

    // ★ NIMが返したレスポンスヘッダー
    if (error.response?.headers) {
      console.error(`  ★ NIM Response Headers:`);
      Object.entries(error.response.headers).forEach(([k, v]) => {
        console.error(`    ${k}: ${v}`);
      });
    }

    console.error('!'.repeat(80) + '\n');

    if (!res.headersSent) {
      // ★ クライアントにもエラー詳細を返す（ブラウザやJanitorAIで見える）
      const errorDetail = {
        error: {
          message: error.message || 'Internal server error',
          type: 'proxy_error',
          code: error.response?.status || 500,
          nim_error: error.response?.data || null,
          debug: {
            nim_status: error.response?.status,
            nim_status_text: error.response?.statusText,
            requested_model: req.body?.model,
            mapped_model: MODEL_MAPPING[req.body?.model] || 'fallback'
          }
        }
      };
      res.status(error.response?.status || 500).json(errorDetail);
    } else {
      res.end();
    }
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log('='.repeat(80));
  console.log('  DIAGNOSTIC MODE - すべてのリクエスト/レスポンスをログに出力します');
  console.log('='.repeat(80));
  console.log(`  Port: ${PORT}`);
  console.log(`  NIM Base: ${NIM_API_BASE}`);
  console.log(`  Thinking mode: ${ENABLE_THINKING_MODE}`);
  console.log(`  Reasoning display: ${SHOW_REASONING}`);
  console.log('='.repeat(80));
});
