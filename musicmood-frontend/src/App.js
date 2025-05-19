import React, { useState, useEffect, useRef } from 'react';
import * as ort from 'onnxruntime-web';
import './App.css';

const MAX_POPULAR_RECOMMENDATIONS = 5;
const MAX_RANDOM_RECOMMENDATIONS = 5;

function App() {
  const [session, setSession] = useState(null);
  const [inputText, setInputText] = useState('');
  const [predictedMoodResult, setPredictedMoodResult] = useState('');
  const [isLoadingModel, setIsLoadingModel] = useState(true);
  const [isLoadingSongData, setIsLoadingSongData] = useState(true);
  const [songData, setSongData] = useState([]);
  const [popularRecommendedSongs, setPopularRecommendedSongs] = useState([]);
  const [randomRecommendedSongs, setRandomRecommendedSongs] = useState([]);
  const [error, setError] = useState('');
  const [infoMessage, setInfoMessage] = useState('Initializing...');

  const onnxRuntimeInitialized = useRef(false);

  useEffect(() => {
    async function loadRessourcen() {
      setIsLoadingModel(true);
      setIsLoadingSongData(true);
      setInfoMessage('Loading AI model...');
      try {
        // ort.env.wasm.wasmPaths = { // Previous local/public path setup
        //   'ort-wasm.wasm': `${process.env.PUBLIC_URL}/onnxruntime-web/dist/ort-wasm.wasm`,
        //   'ort-wasm-simd.wasm': `${process.env.PUBLIC_URL}/onnxruntime-web/dist/ort-wasm-simd.wasm`,
        //   'ort-wasm-threaded.wasm': `${process.env.PUBLIC_URL}/onnxruntime-web/dist/ort-wasm-threaded.wasm`
        // };

        // Set WASM paths to JSDelivr CDN
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

        const modelPath = `${process.env.PUBLIC_URL}/pipeline.onnx`;
        const newSession = await ort.InferenceSession.create(modelPath);
        setSession(newSession);
        console.log("ONNX session created. Inputs:", newSession.inputNames, "Outputs:", newSession.outputNames);
        setInfoMessage('AI Model loaded. Loading song data...');
      } catch (e) {
        console.error(`Failed to load ONNX model: ${e}`);
        setError(`Failed to load AI model: ${e.message}. Please ensure pipeline.onnx is in public/ and wasm files are accessible.`);
        setIsLoadingModel(false);
        setIsLoadingSongData(false);
        setInfoMessage('');
        return;
      }
      setIsLoadingModel(false);

      setInfoMessage('Loading song library...');
      try {
        const dataPath = `${process.env.PUBLIC_URL}/data_moods.json`;
        const response = await fetch(dataPath);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        setSongData(data);
        console.log("Song data loaded.");
        setInfoMessage('Ready to discover your vibe!');
        setTimeout(() => setInfoMessage(''), 3000);
      } catch (e) {
        console.error(`Failed to load song data: ${e}`);
        setError(prevError => prevError ? prevError + ` Failed to load song data: ${e.message}.` : `Failed to load song data: ${e.message}.`);
        setInfoMessage('');
      }
      setIsLoadingSongData(false);
    }

    if (!onnxRuntimeInitialized.current) {
      loadRessourcen();
      onnxRuntimeInitialized.current = true;
    }
  }, []);

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  // Fisher-Yates (Knuth) Shuffle algorithm
  function shuffleArray(array) {
    let currentIndex = array.length,  randomIndex;
    // While there remain elements to shuffle.
    while (currentIndex !== 0) {
      // Pick a remaining element.
      randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex--;
      // And swap it with the current element.
      [array[currentIndex], array[randomIndex]] = [
        array[randomIndex], array[currentIndex]];
    }
    return array;
  }

  const recommendSongsByMood = (mood) => {
    if (!songData || songData.length === 0 || !mood) {
      setPopularRecommendedSongs([]);
      setRandomRecommendedSongs([]);
      return;
    }
    const allMatchingSongs = songData
      .filter(song => song.mood && mood && song.mood.toLowerCase() === mood.toLowerCase());

    // Popular songs
    const popularSongs = [...allMatchingSongs] // Create a copy before sorting
      .sort((a, b) => parseFloat(b.popularity || 0) - parseFloat(a.popularity || 0))
      .slice(0, MAX_POPULAR_RECOMMENDATIONS);
    setPopularRecommendedSongs(popularSongs);

    // Random songs (different from popular ones if possible)
    const popularSongIds = new Set(popularSongs.map(song => song.id)); // Assuming songs have a unique 'id' field
    let remainingSongs = allMatchingSongs.filter(song => !popularSongIds.has(song.id));

    // If not enough unique songs, allow picking from all matching ones for random selection, but try to prioritize different ones
    if (remainingSongs.length < MAX_RANDOM_RECOMMENDATIONS && allMatchingSongs.length > popularSongs.length) {
        // If remaining are too few, but there are more songs than popular ones, use all matching ones for random pool.
        // This increases chance of variety if overlap is forced.
        remainingSongs = allMatchingSongs;
    }

    const shuffledSongs = shuffleArray([...remainingSongs]); // Shuffle a copy
    const randomSongs = shuffledSongs.slice(0, MAX_RANDOM_RECOMMENDATIONS);
    setRandomRecommendedSongs(randomSongs);
  };

  const handlePredict = async () => {
    if (!session) {
      setError("AI Session not loaded. Please wait or refresh."); return;
    }
    if (isLoadingSongData || isLoadingModel) {
      setError("Resources are still loading. Please wait."); return;
    }
    if (!inputText.trim()) {
      setError("Please describe your feeling first!");
      setPredictedMoodResult('');
      setPopularRecommendedSongs([]);
      setRandomRecommendedSongs([]);
      return;
    }

    try {
      setError('');
      setInfoMessage('');
      setPredictedMoodResult('Finding your vibe...');
      setPopularRecommendedSongs([]);
      setRandomRecommendedSongs([]);

      const input = [inputText];
      const tensorInput = new ort.Tensor('string', input, [input.length]);
      const feeds = { [session.inputNames[0]]: tensorInput };
      const actualOutputName = session.outputNames[0];
      const requestedOutputs = [actualOutputName];
      const results = await session.run(feeds, requestedOutputs);
      const outputTensor = results[actualOutputName];

      if (outputTensor && outputTensor.data && outputTensor.data.length > 0) {
        const mood = outputTensor.data[0];
        setPredictedMoodResult(mood);
        recommendSongsByMood(mood);
      } else {
        setError("Couldn't quite catch that vibe. Try rephrasing?");
        setPredictedMoodResult('');
      }
    } catch (e) {
      console.error(`Error during prediction: ${e}`);
      setError(`Prediction error: ${e.message}.`);
      setPredictedMoodResult('');
      setPopularRecommendedSongs([]);
      setRandomRecommendedSongs([]);
    }
  };

  const showLoadingState = isLoadingModel || isLoadingSongData;

  return (
    <div className="App">
      <header className="App-header">
        <h1>Mood Melody AI</h1>

        {showLoadingState && (
          <div className="loading-container">
            <div className="spinner"></div>
            <p className="loading-text">{infoMessage || 'Initializing...'}</p>
          </div>
        )}

        {error && <p className="error-text">{error}</p>}

        {!showLoadingState && session && (
          <div className="input-section">
            <textarea
              rows="4"
              placeholder="How are you feeling today? Let's find some music!"
              value={inputText}
              onChange={handleInputChange}
            />
            <button onClick={handlePredict}>
              Discover Mood & Songs
            </button>
          </div>
        )}

        {!showLoadingState && predictedMoodResult && predictedMoodResult !== 'Finding your vibe...' && (
          <div className="results-container">
            <div className="predicted-mood">
              <h3>Your Vibe: {predictedMoodResult}</h3>
            </div>

            {/* Popular Recommendations */}
            {popularRecommendedSongs.length > 0 && (
              <div className="recommendations-container popular-recommendations">
                <h4>Top Picks for Your Mood:</h4>
                <ul className="song-list">
                  {popularRecommendedSongs.map((song, index) => (
                    <li key={`popular-${song.id || index}`} className="song-item">
                      <strong>{song.name}</strong> by {song.artist}<br />
                      <em>Album: {song.album} (Popularity: {song.popularity})</em>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Random Recommendations */}
            {randomRecommendedSongs.length > 0 && (
              <div className="recommendations-container random-recommendations">
                <h4>More Ideas (Random Picks):</h4>
                <ul className="song-list">
                  {randomRecommendedSongs.map((song, index) => (
                    <li key={`random-${song.id || index}`} className="song-item">
                      <strong>{song.name}</strong> by {song.artist}<br />
                      <em>Album: {song.album} (Popularity: {song.popularity})</em>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {popularRecommendedSongs.length === 0 && randomRecommendedSongs.length === 0 && (
                <p className="no-songs-text">Looks like we couldn't find specific tracks for this vibe in our current list. Try another feeling!</p>
            )}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
