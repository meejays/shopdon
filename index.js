// index.js

require('dotenv').config();

const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error('❌ Missing OPENAI_API_KEY in environment');
  process.exit(1);
}

const OpenAI = require('openai');
const openai = new OpenAI({ apiKey });

/**
 * Sends a user message to the Chat Completions API and returns the assistant's reply.
 * @param {string} userMessage
 * @returns {Promise<string>}
 */
async function getChatResponse(userMessage) {
  const response = await openai.chat.completions.create({
    model: 'gpt-4',
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user',   content: userMessage }
    ]
  });
  return response.choices[0].message.content.trim();
}

// If run directly, demo with a sample prompt
if (require.main === module) {
  getChatResponse('Say hello in a creative way.')
    .then(reply => console.log('🤖 AI says:', reply))
    .catch(err => {
      console.error('❌ OpenAI API error:', err);
      process.exit(1);
    });
}

module.exports = { getChatResponse };
