const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: [{
      id: 'claude-3-sonnet',
      object: 'model',
      created: Date.now(),
      owned_by: 'nvidia-nim-proxy'
    }]
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens } = req.body;

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, {
      model: 'z-ai/glm5',
      messages,
      temperature: temperature || 0.85,
      max_tokens: max_tokens || 9024,
      stream: false,
      chat_template_kwargs: {
        enable_thinking: true,
        clear_thinking: false
      }
    }, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      }
    });

    res.json({
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: model || 'claude-3-sonnet',
      choices: response.data.choices.map(choice => {
        let content = choice.message?.content || '';
        if (choice.message?.reasoning_content) {
          content = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + content;
        }
        return {
          index: choice.index,
          message: { role: choice.message.role, content },
          finish_reason: choice.finish_reason
        };
      }),
      usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    });

  } catch (error) {
    console.error('Proxy error:', error.message);
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

app.listen(PORT, () => {
  console.log(`Proxy running on port ${PORT}`);
});
