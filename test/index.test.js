// File: test/index.test.js

// Mock the OpenAI client class
jest.mock('openai', () => {
  return jest.fn().mockImplementation(() => ({
    chat: {
      completions: {
        create: jest.fn()
      }
    }
  }));
});

const OpenAI = require('openai');
const { getChatResponse } = require('../index');

describe('getChatResponse', () => {
  const fakeReply = '🌟 Hello there, star traveler!';

  beforeAll(() => {
    // Stub the create() method to return our fake structure
    OpenAI.mock.instances[0].chat.completions.create.mockResolvedValue({
      choices: [{ message: { content: ` ${fakeReply} ` } }]
    });
  });

  it('returns the trimmed assistant reply', async () => {
    const reply = await getChatResponse('Any prompt');
    expect(reply).toBe(fakeReply);
    // Verify the API was called with the right params
    expect(OpenAI.mock.instances[0].chat.completions.create).toHaveBeenCalledWith(
      expect.objectContaining({
        model: 'gpt-4',
        messages: expect.any(Array)
      })
    );
  });
});
