// Configuration for the RAG Chatbot
const chatbotConfig = {
  // OpenAI API Configuration
  openai: {
    apiKey: process.env.OPENAI_API_KEY,
    model: 'gpt-4-turbo', // or gpt-3.5-turbo for cost efficiency
  },

  // Qdrant Vector Database Configuration
  qdrant: {
    url: process.env.QDRANT_URL || 'https://your-cluster-url.qdrant.io',
    apiKey: process.env.QDRANT_API_KEY,
    collectionName: 'book_content',
  },

  // Neon Postgres Configuration
  neon: {
    connectionString: process.env.NEON_DB_URL,
  },

  // FastAPI Backend Configuration
  fastapi: {
    baseUrl: process.env.FASTAPI_BASE_URL || 'http://localhost:8000',
  },

  // Chatbot Settings
  chatbot: {
    name: 'Physical AI & Humanoid Robotics Assistant',
    systemPrompt: `You are an expert assistant for the "Physical AI & Humanoid Robotics" book.
    Answer questions based strictly on the book's content.
    Only provide information that can be found in the book.
    If a question cannot be answered based on the book content, politely explain that the information is not available in the book.`,
    maxTokens: 1000,
    temperature: 0.3,
  },

  // Content Retrieval Settings
  retrieval: {
    topK: 5, // Number of chunks to retrieve
    similarityThreshold: 0.7, // Minimum similarity score
    maxChunkSize: 1000, // Maximum size of text chunks in characters
  }
};

module.exports = chatbotConfig;