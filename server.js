// server.js - OpenAI to NVIDIA NIM API Proxy
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

// Fail fast if API key is missing
if (!NIM_API_KEY) {
  console.error('FATAL: NIM_API_KEY environment variable is not set');
  process.exit(1);
}

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = true;

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = true;

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'z-ai/glm4.7',
  'claude-3-sonnet':'z-ai/glm5',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

// Cache for verified direct-passthrough models
const verifiedModels = new Map();

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // Input validation
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: {
          message: "'messages' is required and must be a non-empty array",
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];

    if (!nimModel) {
      // Check cache first
      if (verifiedModels.has(model)) {
        const cached = verifiedModels.get(model);
        if (cached) {
          nimModel = cached;
        }
      } else {
        // Probe whether NIM accepts this model name directly
        try {
          const probeResponse = await axios.post(
            `${NIM_API_BASE}/chat/completions`,
            {
              model: model,
              messages: [{ role: 'user', content: 'test' }],
              max_tokens: 1
            },
            {
              headers: {
                'Authorization': `Bearer ${NIM_API_KEY}`,
                'Content-Type': 'application/json'
              },
              timeout: 10000,
              validateStatus: (status) => status < 500
            }
          );

          if (probeResponse.status >= 200 && probeResponse.status < 300) {
            nimModel = model;
            verifiedModels.set(model, model);
          } else {
            verifiedModels.set(model, null);
          }
        } catch (e) {
          console.warn('Model probe failed:', e.message);
          verifiedModels.set(model, null);
        }
      }

      // Heuristic fallback if probe failed and no mapping exists
      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }

    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.85,
      max_tokens: max_tokens || 9024,
      stream: stream || true,
      ...(ENABLE_THINKING_MODE && { chat_template_kwargs: { thinking: true } })
    };

    // Make request to NVIDIA NIM API
    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        timeout: 120000,
        responseType: stream ? 'stream' : 'json'
      }
    );

    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;

          if (line.includes('[DONE]')) {
            // Close any open reasoning tag before finishing
            if (SHOW_REASONING && reasoningStarted) {
              const closeData = {
                choices: [{
                  index: 0,
                  delta: { content: '</think>\n\n' }
                }]
              };
              res.write(`data: ${JSON.stringify(closeData)}\n\n`);
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

                if (reasoning && !reasoningStarted) {
                  combinedContent = '<think>\n' + reasoning;
                  reasoningStarted = true;
                } else if (reasoning) {
                  combinedContent = reasoning;
                }

                if (content && reasoningStarted) {
                  combinedContent += '</think>\n\n' + content;
                  reasoningStarted = false;
                } else if (content) {
                  combinedContent += content;
                }

                if (combinedContent) {
                  data.choices[0].delta.content = combinedContent;
                }
              } else {
                // When reasoning display is off, pass through content only
                data.choices[0].delta.content = content || '';
              }

              delete data.choices[0].delta.reasoning_content;
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            // Forward unparseable lines as-is
            res.write(line + '\n\n');
          }
        });
      });

      response.data.on('end', () => {
        // Safety net: close reasoning if stream ended without [DONE]
        if (SHOW_REASONING && reasoningStarted) {
          const closeData = {
            choices: [{
              index: 0,
              delta: { content: '</think>\n\n' }
            }]
          };
          res.write(`data: ${JSON.stringify(closeData)}\n\n`);
        }
        res.end();
      });

      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        // Attempt to send an error event before closing
        try {
          const errorData = {
            error: {
              message: 'Upstream stream error: ' + err.message,
              type: 'server_error'
            }
          };
          res.write(`data: ${JSON.stringify(errorData)}\n\n`);
        } catch (_) { /* response may already be closed */ }
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
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

          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }
  } catch (error) {
    console.error('Proxy error:', error.message);

    // Avoid writing JSON to an already-started stream
    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: error.message || 'Internal server error',
          type: 'invalid_request_error',
          code: error.response?.status || 500
        }
      });
    } else {
      res.end();
    }
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
