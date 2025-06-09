import express from 'express';
import fetch from 'node-fetch';
import { pipeline } from '@xenova/transformers';
import cors from 'cors';

const app = express();
app.use(cors());
const port = 3001;

let encoder;

// Initialize encoder on server startup
const init = async () => {
  console.log('Loading embedding model...');
  encoder = await pipeline('feature-extraction', 'Xenova/bge-small-en');
  console.log('✅ Model ready');
};

// Cosine similarity function
const cosine = (a, b) => {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const normA = Math.sqrt(a.reduce((s, ai) => s + ai ** 2, 0));
  const normB = Math.sqrt(b.reduce((s, bi) => s + bi ** 2, 0));
  return dot / (normA * normB);
};

// Fetch Google autocomplete suggestions
const fetchGoogleSuggestions = async (query) => {
  const url = `https://suggestqueries.google.com/complete/search?client=firefox&q=${encodeURIComponent(query)};`
  const res = await fetch(url);
  const data = await res.json();
  return data[1]; // Suggestions list
};

app.get('/suggest', async (req, res) => {
  const q = req.query.q;
  if (!q) return res.status(400).json({ error: 'Missing ?q= parameter' });

  try {
    const rawSuggestions = await fetchGoogleSuggestions(q);
    if (!rawSuggestions.length) return res.json([]);

    const flowerReference = await encoder('flower bouquet gift');
    const flowerVec = Array.from(flowerReference.data[0]);

    const suggestionEmbeddings = await Promise.all(
      rawSuggestions.map(s => encoder(s).then(e => Array.from(e.data[0])))
    );

    let scored = rawSuggestions.map((title, i) => ({
      title,
      score: cosine(flowerVec, suggestionEmbeddings[i])
    }));

    scored = scored.sort((a, b) => b.score - a.score);
    const result = scored.filter(r => r.score > 0.3);

    res.json(result.length ? result.slice(0, 5) : scored.slice(0, 5));
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

init().then(() => {
  app.listen(port, () => {
    console.log(`🚀 Server running at http://localhost:${port}/suggest?q=rose`);
  });
});
