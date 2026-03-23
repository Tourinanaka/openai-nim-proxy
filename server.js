const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// ── Clean history before sending to API ─────────────────
function prepareMessages(messages) {
  return messages.map(msg => {
    let content = typeof msg.content === 'string' ? msg.content : '';
    content = content.replace(/<think>[\s\S]*?<\/think>\s*/g, '').trim();
    return { role: msg.role, content: content || '(continue)' };
  });
}

// ── Line break engine (ported from streaming version) ───
function needsBreak(a, sep, c) {
  if (sep === ' ') {
    if (/[.!?]/.test(a) && c === '"') return true;
    if (/[.!?"'*]/.test(a) && c === '*') return true;
    if (a === '*' && /[A-Z"]/.test(c)) return true;
    if (/["']/.test(a) && /[A-Z]/.test(c)) return true;
  }
  if (sep === '\n') {
    if (a !== '\n' && (c === '"' || c === '*')) return true;
    if (/["']/.test(a) && /[A-Z*]/.test(c)) return true;
  }
  return false;
}

function enforceLineBreaks(text) {
  text = text.replace(/\r\n/g, '\n');
  if (text.length < 3) return text;

  // 3-char sliding window — same logic as streaming version
  let raw = '';
  let i = 0;
  while (i < text.length - 2) {
    if (needsBreak(text[i], text[i + 1], text[i + 2])) {
      raw += text[i] + '\n\n';
      i += 2;   // skip the separator, next loop starts at char c
    } else {
      raw += text[i];
      i++;
    }
  }
  // append remaining 1–2 chars that couldn't form a full window
  while (i < text.length) {
    raw += text[i];
    i++;
  }

  // collapse 3+ newlines → 2
  let out = '';
  let nlRun = 0;
  for (const ch of raw) {
    if (ch === '\n') {
      nlRun++;
      if (nlRun <= 2) out += '\n';
    } else {
      nlRun = 0;
      out += ch;
    }
  }

  return out;
}

// ── Routes ──────────────────────────────────────────────

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
    const preparedMessages = prepareMessages(messages);

    console.log('\n── Turn ──');
    console.log(`  Messages in history: ${messages.length}`);
    messages.slice(-2).forEach(m => {
      const preview = (m.content || '').substring(0, 200).replace(/\n/g, '\\n');
      console.log(`  [${m.role}] ${preview}`);
    });

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, {
      model: 'z-ai/glm5',
      messages: preparedMessages,
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

        let thinkBlock = '';
        let body = content;

        const thinkMatch = content.match(/^\s*(<think>[\s\S]*?<\/think>)\s*([\s\S]*)$/);
        if (thinkMatch) {
          thinkBlock = thinkMatch[1];
          body = thinkMatch[2];
        }

        if (!thinkBlock && choice.message?.reasoning_content) {
          thinkBlock = '<think>\n' + choice.message.reasoning_content + '\n</think>';
        }

        body = enforceLineBreaks(body);

        const finalContent = thinkBlock
          ? thinkBlock + '\n\n' + body
          : body;

        console.log(`  [output] len=${body.length} paragraphs=${(body.match(/\n\n/g) || []).length + 1} think=${!!thinkBlock}`);

        return {
          index: choice.index,
          message: { role: choice.message.role, content: finalContent },
          finish_reason: choice.finish_reason
        };
      }),
      usage: response.data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    });

  } catch (error) {
    console.error('Proxy error:', error.response?.data || error.message);
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
