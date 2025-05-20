import React, { useState, useEffect, useRef } from 'react';
import * as ort from 'onnxruntime-web';
import './App.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faMusic, faQuestionCircle, faFileAlt, faChartBar } from '@fortawesome/free-solid-svg-icons';

const MAX_POPULAR_RECOMMENDATIONS = 5;
const MAX_RANDOM_RECOMMENDATIONS = 5;
const NUM_ANIMATED_NOTES = 15;
// const NOTE_TYPES = ['whole', 'half', 'quarter', 'eighth']; // Define note types

const MOOD_THEMES = {
  default: {
    emoji: '🎶',
    colors: {
      primary: 'linear-gradient(to right, #3b82f6, #2563eb)',
      primaryHover: '#1d4ed8', // Darker solid for hover when primary is gradient
      background: 'linear-gradient(to bottom right, #f0f4f8, #e0e7ff)',
      cardBackground: '#ffffff',
      textPrimary: '#1f2937',
      textSecondary: '#4b5563',
      borderColor: '#d1d5db',
    }
  },
  happy: {
    emoji: '😊',
    colors: {
      primary: 'linear-gradient(to right, #10b981, #059669)', // Green gradient
      primaryHover: '#047857', // Darker solid green
      background: 'linear-gradient(to bottom right, #ecfdf5, #a7f3d0)', // Light green to slightly more saturated green
      cardBackground: '#ffffff',
      textPrimary: '#065f46',
      textSecondary: '#047857',
      borderColor: '#6ee7b7', // Brighter green border
    }
  },
  sad: {
    emoji: '😢',
    colors: {
      primary: 'linear-gradient(to right, #60a5fa, #3b82f6)', // Blue gradient
      primaryHover: '#2563eb', // Darker solid blue
      background: 'linear-gradient(to bottom right, #eef2ff, #dbeafe)', // Light indigo/blue to slightly darker
      cardBackground: '#f0f4f8',
      textPrimary: '#374151',
      textSecondary: '#4b5563',
      borderColor: '#a5b4fc', // Lighter indigo border
    }
  },
  energetic: {
    emoji: '⚡',
    colors: {
      primary: 'linear-gradient(to right, #ec4899, #db2777)', // Pink gradient
      primaryHover: '#be185d', // Darker solid pink
      background: 'linear-gradient(to bottom right, #fdf2f8, #fce7f3)', // Light pink to slightly more saturated
      cardBackground: '#ffffff',
      textPrimary: '#831843',
      textSecondary: '#be185d',
      borderColor: '#f9a8d4', // Brighter pink border
    }
  },
  calm: {
    emoji: '😌',
    colors: {
      primary: 'linear-gradient(to right, #22c55e, #16a34a)', // Another green gradient for calm
      primaryHover: '#15803d', // Darker solid green
      background: 'linear-gradient(to bottom right, #f0fdf4, #ccfbf1)', // Light teal/green to slightly more saturated
      cardBackground: '#ffffff',
      textPrimary: '#14532d',
      textSecondary: '#15803d',
      borderColor: '#5eead4', // Brighter teal/green border
    }
  },
  angry: {
    emoji: '😠',
    colors: {
      primary: 'linear-gradient(to right, #ef4444, #dc2626)', // Red gradient
      primaryHover: '#b91c1c', // Darker solid red
      background: 'linear-gradient(to bottom right, #fee2e2, #fecaca)', // Light red to slightly more saturated
      cardBackground: '#fef2f2',
      textPrimary: '#7f1d1d',
      textSecondary: '#b91c1c',
      borderColor: '#fca5a5', // Brighter red border
    }
  },
  // Add other moods your model might predict, e.g., relaxed, romantic, etc.
  // Ensure keys are lowercase for easier matching.
};

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
  const [showHelp, setShowHelp] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [showStats, setShowStats] = useState(false);
  const [animatedNotes, setAnimatedNotes] = useState([]); // State for animated notes
  const [currentMoodKey, setCurrentMoodKey] = useState('default'); // State for current mood theme key

  const onnxRuntimeInitialized = useRef(false);

  useEffect(() => {
    async function loadRessourcen() {
      setIsLoadingModel(true);
      setIsLoadingSongData(true);
      setInfoMessage('Loading AI model...');
      try {
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

    // Generate animated notes on mount
    const notesArray = [];
    for (let i = 0; i < NUM_ANIMATED_NOTES; i++) {
      // const type = NOTE_TYPES[Math.floor(Math.random() * NOTE_TYPES.length)]; // Assign random type
      notesArray.push({
        id: i,
        type: 'eighth', // Store the type
        left: `${Math.random() * 100}%`,
        top: `${Math.random() * 30 + 70}%`,
        animationDuration: `${Math.random() * 5 + 7}s`,
        animationDelay: `${Math.random() * 10}s`,
        size: `${Math.random() * 8 + 8}px`, // Whole notes slightly larger base size // Simplified size
        randomXs: Math.random() * 2 - 1,
        randomXe: Math.random() * 4 - 2,
      });
    }
    setAnimatedNotes(notesArray);

  }, []); // Empty dependency array ensures this runs only once on mount

  useEffect(() => {
    const moodTheme = MOOD_THEMES[currentMoodKey.toLowerCase()] || MOOD_THEMES.default;
    const colors = moodTheme.colors;
    document.documentElement.style.setProperty('--primary-color', colors.primary);
    document.documentElement.style.setProperty('--primary-hover-color', colors.primaryHover);
    document.documentElement.style.setProperty('--background-color', colors.background);
    document.documentElement.style.setProperty('--card-background-color', colors.cardBackground);
    document.documentElement.style.setProperty('--text-primary-color', colors.textPrimary);
    document.documentElement.style.setProperty('--text-secondary-color', colors.textSecondary);
    document.documentElement.style.setProperty('--border-color', colors.borderColor);
    // Error colors are kept constant for now, but could also be themed if desired
  }, [currentMoodKey]);

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  function shuffleArray(array) {
    let currentIndex = array.length,  randomIndex;
    while (currentIndex !== 0) {
      randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex--;
      [array[currentIndex], array[randomIndex]] = [array[randomIndex], array[currentIndex]];
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

    const popularSongs = [...allMatchingSongs]
      .sort((a, b) => parseFloat(b.popularity || 0) - parseFloat(a.popularity || 0))
      .slice(0, MAX_POPULAR_RECOMMENDATIONS);
    setPopularRecommendedSongs(popularSongs);

    const popularSongIds = new Set(popularSongs.map(song => song.id));
    let remainingSongs = allMatchingSongs.filter(song => !popularSongIds.has(song.id));

    if (remainingSongs.length < MAX_RANDOM_RECOMMENDATIONS && allMatchingSongs.length > popularSongs.length) {
        remainingSongs = allMatchingSongs;
    }

    const shuffledSongs = shuffleArray([...remainingSongs]);
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
        const moodKey = mood.toLowerCase();
        setCurrentMoodKey(MOOD_THEMES[moodKey] ? moodKey : 'default');
        recommendSongsByMood(mood);
      } else {
        setError("Couldn't quite catch that vibe. Try rephrasing?");
        setPredictedMoodResult('');
        setCurrentMoodKey('default');
      }
    } catch (e) {
      console.error(`Error during prediction: ${e}`);
      setError(`Prediction error: ${e.message}.`);
      setPredictedMoodResult('');
      setCurrentMoodKey('default');
      setPopularRecommendedSongs([]);
      setRandomRecommendedSongs([]);
    }
  };

  const showLoadingState = isLoadingModel || isLoadingSongData;

  const toggleHelp = () => {
    setShowHelp(!showHelp);
    setShowLogs(false);
    setShowStats(false);
  };

  const toggleLogs = () => {
    setShowLogs(!showLogs);
    setShowHelp(false);
    setShowStats(false);
  };

  const toggleStats = () => {
    setShowStats(!showStats);
    setShowHelp(false);
    setShowLogs(false);
  };

  const activeTheme = MOOD_THEMES[currentMoodKey.toLowerCase()] || MOOD_THEMES.default;

  return (
    <div className="App">
      <div className="musical-notes">
        {animatedNotes.map(note => (
          <div
            key={note.id}
            className="note"
            data-note-type={note.type}
            style={{
              left: note.left,
              top: note.top,
              width: note.size,
              height: note.size,
              animationDuration: note.animationDuration,
              animationDelay: note.animationDelay,
              '--random-xs': note.randomXs,
              '--random-xe': note.randomXe,
            }}
          />
        ))}
      </div>
      <header className="App-header">
        <div className="logo-title">
            <FontAwesomeIcon icon={faMusic} size="2x" />
            <h1>Mood Melody AI</h1>
        </div>

        {showLoadingState && (
          <div className="loading-container">
            <div className="spinner"></div>
            <p className="loading-text">{infoMessage || 'Initializing...'}</p>
          </div>
        )}

        {error && <p className="error-text">{error}</p>}
        
        <div className="control-buttons-container">
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
            <div className="control-buttons">
              <button onClick={toggleHelp} className="control-button help-button" title="Help">
                <FontAwesomeIcon icon={faQuestionCircle} />
              </button>
              <button onClick={toggleLogs} className="control-button logs-button" title="Show Logs">
                <FontAwesomeIcon icon={faFileAlt} />
              </button>
              <button onClick={toggleStats} className="control-button stats-button" title="Show Inference Stats">
                <FontAwesomeIcon icon={faChartBar} />
              </button>
            </div>
        </div>
      </header>

      {!showLoadingState && predictedMoodResult && predictedMoodResult !== 'Finding your vibe...' && (
        <div className="results-container">
          <div className="predicted-mood">
            <h3>Your Vibe: {predictedMoodResult} {activeTheme.emoji}</h3>
          </div>
          {popularRecommendedSongs.length > 0 && (
            <div className="recommendations-container popular-recommendations">
              <h4>Top Picks for Your Mood:</h4>
              <ul className="song-list">
                {popularRecommendedSongs.map((song, index) => (
                  <li key={`popular-${song.id || index}`} className="song-item">
                    <a href={`https://open.spotify.com/track/${song.id}`} target="_blank" rel="noopener noreferrer">
                      <strong>{song.name}</strong> by {song.artist}
                    </a>
                    <br />
                    <em>Album: {song.album} (Popularity: {song.popularity})</em>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {randomRecommendedSongs.length > 0 && (
            <div className="recommendations-container random-recommendations">
              <h4>More Ideas (Random Picks):</h4>
              <ul className="song-list">
                {randomRecommendedSongs.map((song, index) => (
                  <li key={`random-${song.id || index}`} className="song-item">
                    <a href={`https://open.spotify.com/track/${song.id}`} target="_blank" rel="noopener noreferrer">
                      <strong>{song.name}</strong> by {song.artist}
                    </a>
                    <br />
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

      {showHelp && (
        <div className="modal-overlay">
          <div className="modal-content help-modal">
            <h2><FontAwesomeIcon icon={faQuestionCircle} /> Help</h2>
            <p>This application uses an AI to predict your mood based on text and then recommends songs from a predefined library that match that mood.</p>
            <h3>How to Use:</h3>
            <ul>
              <li><strong>Describe Your Feeling:</strong> Type how you're feeling into the text area.</li>
              <li><strong>Discover:</strong> Click "Discover Mood & Songs".</li>
              <li><strong>Recommendations:</strong> The AI will show your predicted mood and suggest songs.</li>
              <li><strong>Spotify Links:</strong> Click on a song to open it in Spotify.</li>
            </ul>
            <button onClick={toggleHelp}>Close</button>
          </div>
        </div>
      )}

      {showLogs && (
        <div className="logs-container modal-overlay">
          <div className="modal-content logs-modal">
            <h3><FontAwesomeIcon icon={faFileAlt} /> Inference Logs</h3>
            <pre className="logs-pre">
              Log entry 1: Model initialized successfully.
              Log entry 2: Input text received: "Happy and energetic"
              Log entry 3: Preprocessing input...
              Log entry 4: Running inference with ONNX model...
              Log entry 5: Inference completed in 0.05s.
              Log entry 6: Predicted mood: Joyful
              Log entry 7: Fetching song recommendations for Joyful...
              Log entry 8: Displaying recommendations.
            </pre>
            <button onClick={toggleLogs}>Close Logs</button>
          </div>
        </div>
      )}

      {showStats && (
        <div className="stats-container modal-overlay">
          <div className="modal-content stats-modal">
            <h3><FontAwesomeIcon icon={faChartBar} /> Inference Statistics</h3>
            <pre className="stats-pre">
              Probability for Mood A: 0.75
              Probability for Mood B: 0.15
              Probability for Mood C: 0.05
              Other Probabilities: ...
              Confidence Score: 0.88
              Key Features Used: [FeatureX, FeatureY]
            </pre>
            <button onClick={toggleStats}>Close Stats</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
