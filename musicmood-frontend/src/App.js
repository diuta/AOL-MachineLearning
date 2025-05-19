import React, { useState, useEffect, useRef } from 'react';
import * as ort from 'onnxruntime-web';
import './App.css';

const MAX_RECOMMENDATIONS = 5; // Max number of songs to recommend

function App() {
  const [session, setSession] = useState(null);
  const [inputText, setInputText] = useState('');
  const [predictedMoodResult, setPredictedMoodResult] = useState(''); // Stores the actual predicted mood string
  const [isLoadingModel, setIsLoadingModel] = useState(true);
  const [isLoadingSongData, setIsLoadingSongData] = useState(true);
  const [songData, setSongData] = useState([]);
  const [recommendedSongs, setRecommendedSongs] = useState([]);
  const [error, setError] = useState('');
  const [infoMessage, setInfoMessage] = useState('Initializing...'); // For general info like loading

  const onnxRuntimeInitialized = useRef(false);

  useEffect(() => {
    async function loadRessourcen() {
      setIsLoadingModel(true);
      setIsLoadingSongData(true);
      setInfoMessage('Loading AI model...');
      try {
        ort.env.wasm.wasmPaths = {
          'ort-wasm.wasm': '/onnxruntime-web/dist/ort-wasm.wasm',
          'ort-wasm-simd.wasm': '/onnxruntime-web/dist/ort-wasm-simd.wasm',
          'ort-wasm-threaded.wasm': '/onnxruntime-web/dist/ort-wasm-threaded.wasm'
        };
        const newSession = await ort.InferenceSession.create('/pipeline.onnx');
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
        const response = await fetch('/data_moods.json');
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

  const recommendSongsByMood = (mood) => {
    if (!songData || songData.length === 0 || !mood) {
      setRecommendedSongs([]);
      return;
    }
    const filteredSongs = songData
      .filter(song => song.mood && mood && song.mood.toLowerCase() === mood.toLowerCase())
      .sort((a, b) => parseFloat(b.popularity || 0) - parseFloat(a.popularity || 0))
      .slice(0, MAX_RECOMMENDATIONS);
    setRecommendedSongs(filteredSongs);
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
      setRecommendedSongs([]);
      return;
    }

    try {
      setError('');
      setInfoMessage(''); // Clear any lingering info messages
      setPredictedMoodResult('Finding your vibe...');
      setRecommendedSongs([]);

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
      setRecommendedSongs([]);
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
            {recommendedSongs.length > 0 ? (
              <div className="recommendations-container">
                <h4>Here are some tunes for your mood:</h4>
                <ul className="song-list">
                  {recommendedSongs.map((song, index) => (
                    <li key={index} className="song-item">
                      <strong>{song.name}</strong> by {song.artist}<br />
                      <em>Album: {song.album} (Popularity: {song.popularity})</em>
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
                <p className="no-songs-text">Looks like we couldn't find specific tracks for this vibe in our current list. Try another feeling!</p>
            )}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
