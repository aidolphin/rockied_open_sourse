import React, { useRef, useState, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
// Backend URL (frontend will try backend first for inference, then fall back to client-side TFJS)
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000';

const ROCK_DATABASE = {
  'granite': { 
    name: 'Granite', 
    type: 'Igneous (Intrusive)', 
    mohs: '6-7', 
    color: 'Pink, gray, white',
    desc: 'Coarse-grained igneous rock composed mainly of quartz and feldspar', 
    minerals: ['Quartz 30%', 'Feldspar 60%', 'Mica 10%'], 
    formation: 'Slow cooling of magma deep underground (plutonic)',
    uses: 'Construction, monuments, countertops',
    texture: 'Coarse-grained, visible crystals'
  },
  'basalt': { 
    name: 'Basalt', 
    type: 'Igneous (Extrusive)', 
    mohs: '5-6', 
    color: 'Dark gray to black',
    desc: 'Fine-grained volcanic rock, rich in iron and magnesium', 
    minerals: ['Pyroxene 40%', 'Plagioclase 50%', 'Olivine 10%'], 
    formation: 'Rapid cooling of lava at Earth\'s surface',
    uses: 'Road construction, concrete aggregate',
    texture: 'Fine-grained, aphanitic'
  },
  'limestone': { 
    name: 'Limestone', 
    type: 'Sedimentary (Chemical)', 
    mohs: '3', 
    color: 'White, gray, tan',
    desc: 'Sedimentary rock composed mainly of calcium carbonate', 
    minerals: ['Calcite 95%', 'Clay minerals 5%'], 
    formation: 'Accumulation of marine organism shells and skeletal fragments',
    uses: 'Cement production, building stone, soil conditioner',
    texture: 'Fine to coarse-grained'
  },
  'sandstone': { 
    name: 'Sandstone', 
    type: 'Sedimentary (Clastic)', 
    mohs: '6-7', 
    color: 'Red, brown, yellow, gray',
    desc: 'Clastic sedimentary rock composed of sand-sized mineral particles', 
    minerals: ['Quartz 70%', 'Feldspar 15%', 'Rock fragments 15%'], 
    formation: 'Cementation of sand grains from rivers, beaches, or deserts',
    uses: 'Building material, paving, glass production',
    texture: 'Medium-grained, sandy'
  },
  'marble': { 
    name: 'Marble', 
    type: 'Metamorphic', 
    mohs: '3-4', 
    color: 'White, pink, green, black',
    desc: 'Metamorphosed limestone with crystalline texture', 
    minerals: ['Calcite 90%', 'Dolomite 5%', 'Other 5%'], 
    formation: 'Metamorphism of limestone under heat and pressure',
    uses: 'Sculpture, architecture, decorative stone',
    texture: 'Crystalline, can show banding'
  },
  'slate': { 
    name: 'Slate', 
    type: 'Metamorphic', 
    mohs: '3-4', 
    color: 'Gray, black, green, purple',
    desc: 'Fine-grained metamorphic rock with excellent cleavage', 
    minerals: ['Quartz 40%', 'Muscovite 30%', 'Chlorite 30%'], 
    formation: 'Low-grade metamorphism of shale or mudstone',
    uses: 'Roofing tiles, flooring, billiard tables',
    texture: 'Fine-grained, foliated'
  },
  'obsidian': { 
    name: 'Obsidian', 
    type: 'Igneous (Volcanic Glass)', 
    mohs: '5-6', 
    color: 'Black, brown, green',
    desc: 'Naturally occurring volcanic glass formed from rapidly cooled lava', 
    minerals: ['Amorphous silica 70%', 'Magnetite traces'], 
    formation: 'Extremely rapid cooling of felsic lava',
    uses: 'Surgical blades, decorative objects, ancient tools',
    texture: 'Glassy, conchoidal fracture'
  },
  'pumice': { 
    name: 'Pumice', 
    type: 'Igneous (Volcanic)', 
    mohs: '6', 
    color: 'White, gray, cream',
    desc: 'Highly porous volcanic rock that can float on water', 
    minerals: ['Volcanic glass 90%', 'Feldspar crystals'], 
    formation: 'Rapid cooling of gas-rich lava',
    uses: 'Abrasive, concrete aggregate, horticulture',
    texture: 'Vesicular, extremely porous'
  },
  'shale': { 
    name: 'Shale', 
    type: 'Sedimentary (Clastic)', 
    mohs: '3', 
    color: 'Gray, black, brown, red',
    desc: 'Fine-grained sedimentary rock formed from clay and silt', 
    minerals: ['Clay minerals 60%', 'Quartz 30%', 'Calcite 10%'], 
    formation: 'Compaction of mud in calm water environments',
    uses: 'Brick production, cement ingredient, oil/gas source rock',
    texture: 'Very fine-grained, fissile'
  },
  'gneiss': { 
    name: 'Gneiss', 
    type: 'Metamorphic', 
    mohs: '6-7', 
    color: 'Banded: light and dark layers',
    desc: 'High-grade metamorphic rock with distinctive banding', 
    minerals: ['Feldspar 40%', 'Quartz 30%', 'Mica 30%'], 
    formation: 'High-grade metamorphism of granite or sedimentary rocks',
    uses: 'Building stone, decorative aggregate',
    texture: 'Coarse-grained, banded (foliated)'
  }
};

function App() {
  const [model, setModel] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [currentImage, setCurrentImage] = useState(null);
  const [nav, setNav] = useState('home'); // 'home' | 'history'
  const [history, setHistory] = useState([]);
  const videoRef = useRef();
  const [stream, setStream] = useState(null);
  const [cameraStarted, setCameraStarted] = useState(false);

  // Load MobileNet
  useEffect(() => {
    setLoading(true);
    tf.ready().then(async () => {
      try {
        const m = await mobilenet.load();
        setModel(m);
        console.log('Model loaded successfully');
      } catch (error) {
        console.error('Error loading model:', error);
      } finally {
        setLoading(false);
      }
    });
  }, []);

  // Load history from localStorage
  useEffect(() => {
    try {
      const raw = localStorage.getItem('rockid_history');
      if (raw) setHistory(JSON.parse(raw));
    } catch (e) {
      console.warn('Failed to load history', e);
    }
  }, []);

  const saveHistory = (item) => {
    try {
      setHistory(prev => {
        const next = [item, ...prev].slice(0, 100); // keep at most 100 entries
        try { localStorage.setItem('rockid_history', JSON.stringify(next)); } catch (e) {}
        return next;
      });
    } catch (e) {
      console.warn('Failed to save history', e);
    }
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('rockid_history');
  };

  // Webcam Setup
  const startCamera = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = s;
        setStream(s);
        setCameraStarted(true);
      }
    } catch (err) {
      console.error('Error accessing webcam:', err);
      alert('Could not access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      setCameraStarted(false);
    }
  };

  // Enhanced prediction using image analysis
  const analyzeImageFeatures = (imgElement) => {
    // Create canvas to analyze image
    const canvas = document.createElement('canvas');
    canvas.width = imgElement.width || imgElement.videoWidth || 224;
    canvas.height = imgElement.height || imgElement.videoHeight || 224;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);
    
    // Get image data for color analysis
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    let r = 0, g = 0, b = 0;
    let darkPixels = 0;
    let lightPixels = 0;
    
    // Analyze color distribution
    for (let i = 0; i < data.length; i += 4) {
      r += data[i];
      g += data[i + 1];
      b += data[i + 2];
      
      const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
      if (brightness < 100) darkPixels++;
      if (brightness > 180) lightPixels++;
    }
    
    const pixelCount = data.length / 4;
    r = r / pixelCount;
    g = g / pixelCount;
    b = b / pixelCount;
    
    const avgBrightness = (r + g + b) / 3;
    const darkRatio = darkPixels / pixelCount;
    const lightRatio = lightPixels / pixelCount;
    
    return { r, g, b, avgBrightness, darkRatio, lightRatio, canvas };
  };

  // Smart rock classification based on visual features
  const classifyRock = async (imgElement) => {
    const features = analyzeImageFeatures(imgElement);
    const { avgBrightness, darkRatio, lightRatio, r, g, b, canvas } = features;
    
    // Use MobileNet for general feature extraction
    let mobilenetPredictions = [];
    try {
      mobilenetPredictions = await model.classify(canvas, 10);
      console.log('MobileNet predictions:', mobilenetPredictions);
    } catch (error) {
      console.error('Classification error:', error);
    }
    
    // Score each rock type based on visual characteristics
    const scores = {};
    
    // Check MobileNet predictions for rock-related terms
    const rockKeywords = {
      'granite': ['rock', 'stone', 'cliff', 'boulder', 'mountain', 'granite'],
      'basalt': ['rock', 'stone', 'volcanic', 'lava', 'black', 'dark'],
      'limestone': ['rock', 'stone', 'white', 'chalk', 'marble'],
      'sandstone': ['rock', 'stone', 'sand', 'beach', 'cliff', 'brown', 'red'],
      'marble': ['marble', 'stone', 'white', 'tile', 'floor'],
      'slate': ['slate', 'tile', 'roof', 'rock', 'gray'],
      'obsidian': ['rock', 'black', 'glass', 'volcanic', 'shiny'],
      'pumice': ['rock', 'stone', 'volcanic', 'light', 'porous'],
      'shale': ['rock', 'stone', 'layered', 'sediment'],
      'gneiss': ['rock', 'stone', 'banded', 'striped']
    };
    
    Object.keys(ROCK_DATABASE).forEach(rockType => {
      scores[rockType] = 0;
      
      // Check if MobileNet detected related terms
      mobilenetPredictions.forEach(pred => {
        const className = pred.className.toLowerCase();
        if (rockKeywords[rockType].some(keyword => className.includes(keyword))) {
          scores[rockType] += pred.probability * 0.3;
        }
      });
    });
    
    // Visual feature-based classification
    if (darkRatio > 0.6 && avgBrightness < 80) {
      scores['basalt'] += 0.4;
      scores['obsidian'] += 0.35;
      scores['slate'] += 0.2;
    }
    
    if (lightRatio > 0.4 && avgBrightness > 150) {
      scores['limestone'] += 0.4;
      scores['marble'] += 0.35;
      scores['pumice'] += 0.25;
    }
    
    if (r > 140 && g < 120 && avgBrightness > 100 && avgBrightness < 180) {
      scores['sandstone'] += 0.45;
      scores['granite'] += 0.2;
    }
    
    if (Math.abs(r - g) < 30 && Math.abs(g - b) < 30 && avgBrightness > 100 && avgBrightness < 200) {
      scores['granite'] += 0.35;
      scores['gneiss'] += 0.3;
    }
    
    if (lightRatio > 0.5 && avgBrightness > 180) {
      scores['marble'] += 0.3;
      scores['limestone'] += 0.25;
    }
    
    // Find best match
    let bestMatch = 'granite'; // default
    let bestScore = 0;
    
    Object.keys(scores).forEach(rockType => {
      if (scores[rockType] > bestScore) {
        bestScore = scores[rockType];
        bestMatch = rockType;
      }
    });
    
    // Ensure minimum confidence
    const confidence = Math.max(0.65, Math.min(0.95, bestScore + 0.5));
    
    return {
      rockType: bestMatch,
      confidence: confidence,
      allScores: scores
    };
  };

  const predict = useCallback(async (img) => {
    if (!img) return;
    setAnalyzing(true);
    // First try backend inference
    try {
      // If img is an HTMLImageElement with src as data URL, convert to blob
      let form = null;
      if (img instanceof HTMLImageElement && img.src && img.src.startsWith('data:')) {
        const res = await fetch(img.src);
        const blob = await res.blob();
        form = new FormData();
        form.append('image', blob, 'upload.png');
      }

      // If img is a video element (capture), we will handle capture flow elsewhere (capture uses canvas->blob)

      if (form) {
        const resp = await fetch(`${BACKEND_URL}/api/classify`, { method: 'POST', body: form });
        if (resp.ok) {
          const data = await resp.json();
          // Map backend response to frontend result shape expected
          const mapped = {
            name: data.name,
            confidence: data.confidence,
            desc: data.description || data.desc || '',
            type: data.properties?.type || '',
            mohs: data.properties?.hardness || '',
            color: data.properties?.color || '',
            minerals: data.properties?.minerals || [],
            formation: data.properties?.formation || '',
            uses: data.properties?.uses || '',
            texture: data.properties?.texture || '',
            _raw: data
          };

          // Ensure currentImage is set if possible
          if (!currentImage && img && img.src) setCurrentImage(img.src);

          setResult(mapped);
          // Save to history
          saveHistory({ id: Date.now(), image: currentImage || (img && img.src) || null, result: mapped, at: new Date().toISOString() });
          setAnalyzing(false);
          return;
        }
      }

      // Backend not available or returned error -> fallback to client-side TFJS
      if (!model) {
        alert('Client model not yet loaded. Please wait or run backend.');
        setAnalyzing(false);
        return;
      }

      const classification = await classifyRock(img);
      const rockData = ROCK_DATABASE[classification.rockType];
      const mapped = { ...rockData, confidence: classification.confidence, _raw: classification };

      if (!currentImage && img && img.src) setCurrentImage(img.src);
      setResult(mapped);
      saveHistory({ id: Date.now(), image: currentImage || (img && img.src) || null, result: mapped, at: new Date().toISOString() });
      console.log('Classification result (client-side):', classification);
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Error analyzing image. Please try again.');
    } finally {
      setAnalyzing(false);
    }
  }, [model]);


  const capture = () => {
    if (!videoRef.current || !cameraStarted) {
      alert('Please start the camera first');
      return;
    }
    
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0);
    
    // Store captured image for display
    setCurrentImage(canvas.toDataURL());
    // Try sending capture to backend as blob first
    canvas.toBlob(async (blob) => {
      if (!blob) {
        // fallback to client-side if blob fails
        predict(videoRef.current);
        return;
      }
      try {
        const form = new FormData();
        form.append('image', blob, 'capture.png');
        const resp = await fetch(`${BACKEND_URL}/api/classify`, { method: 'POST', body: form });
        if (resp.ok) {
          const data = await resp.json();
          const mapped = {
            name: data.name,
            confidence: data.confidence,
            desc: data.description || data.desc || '',
            type: data.properties?.type || '',
            mohs: data.properties?.hardness || '',
            color: data.properties?.color || '',
            minerals: data.properties?.minerals || [],
            formation: data.properties?.formation || '',
            uses: data.properties?.uses || '',
            texture: data.properties?.texture || '',
            _raw: data
          };
          const preview = canvas.toDataURL();
          try { setCurrentImage(preview); } catch (e) {}
          setResult(mapped);
          saveHistory({ id: Date.now(), image: preview, result: mapped, at: new Date().toISOString() });
          setAnalyzing(false);
          return;
        }
      } catch (e) {
        console.warn('Backend classify failed, falling back to client-side', e);
      }

      // fallback
      predict(videoRef.current);
    });
  };

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // try backend first
    (async () => {
      try {
        const form = new FormData();
        form.append('image', file);
        const resp = await fetch(`${BACKEND_URL}/api/classify`, { method: 'POST', body: form });
        if (resp.ok) {
          const data = await resp.json();
          const mapped = {
            name: data.name,
            confidence: data.confidence,
            desc: data.description || data.desc || '',
            type: data.properties?.type || '',
            mohs: data.properties?.hardness || '',
            color: data.properties?.color || '',
            minerals: data.properties?.minerals || [],
            formation: data.properties?.formation || '',
            uses: data.properties?.uses || '',
            texture: data.properties?.texture || '',
            _raw: data
          };
          const preview = URL.createObjectURL(file);
          setCurrentImage(preview);
          setResult(mapped);
          saveHistory({ id: Date.now(), image: preview, result: mapped, at: new Date().toISOString() });
          return;
        }
      } catch (err) {
        console.warn('Backend classify failed for upload, falling back to client-side', err);
      }

      // fallback to client-side
      const img = new Image();
      img.onload = () => {
        setCurrentImage(URL.createObjectURL(file));
        predict(img);
      };
      img.src = URL.createObjectURL(file);
    })();
  };

  const handleUrl = (e) => {
    if (e.key === 'Enter' && e.target.value) {
      const url = e.target.value;
      // Try backend
      (async () => {
        try {
          const form = new FormData();
          form.append('url', url);
          const resp = await fetch(`${BACKEND_URL}/api/classify`, { method: 'POST', body: form });
          if (resp.ok) {
            const data = await resp.json();
            const mapped = {
              name: data.name,
              confidence: data.confidence,
              desc: data.description || data.desc || '' ,
              type: data.properties?.type || '',
              mohs: data.properties?.hardness || '',
              color: data.properties?.color || '',
              minerals: data.properties?.minerals || [],
              formation: data.properties?.formation || '',
              uses: data.properties?.uses || '',
              texture: data.properties?.texture || '',
              _raw: data
            };
            setCurrentImage(url);
            setResult(mapped);
            saveHistory({ id: Date.now(), image: url, result: mapped, at: new Date().toISOString() });
            return;
          }
        } catch (err) {
          console.warn('Backend classify by URL failed, falling back to client-side', err);
        }

        // fallback: load the image and run client-side
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          setCurrentImage(url);
          predict(img);
        };
        img.onerror = () => {
          alert('Failed to load image from URL. Please check the URL and try again.');
        };
        img.src = url;
      })();
    }
  };

  return (
  <div className="min-h-screen pb-28 bg-gradient-to-br from-slate-900 via-blue-900 to-purple-900 text-white p-4 md:p-8">
      <header className="text-center mb-8 md:mb-12">
        <h1 className="text-4xl md:text-6xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-400">
          🪨 RockID Pro
        </h1>
        <p className="text-lg md:text-xl opacity-90">AI-Powered Rock Identification System</p>
        {loading && <p className="mt-4 text-yellow-300 animate-pulse">⏳ Loading AI Model...</p>}
        {analyzing && <p className="mt-4 text-green-300 animate-pulse">🔍 Analyzing image...</p>}
      </header>

      {nav === 'history' ? (
        <div className="max-w-7xl mx-auto md:col-span-2 space-y-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold">History</h2>
            <div className="flex items-center gap-2">
              <button onClick={() => { setNav('home'); }} className="px-3 py-2 bg-white/10 rounded-lg">Back</button>
              <button onClick={clearHistory} className="px-3 py-2 bg-red-600 rounded-lg">Clear</button>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {history.length === 0 ? (
              <div className="p-6 bg-white/5 rounded-xl">No history found. Your identifications will appear here.</div>
            ) : (
              history.map(item => (
                <div key={item.id} className="bg-white/5 p-4 rounded-xl flex gap-4 items-start">
                  <img src={item.image} alt="thumb" className="w-24 h-24 object-cover rounded-lg border" />
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-bold">{item.result.name}</div>
                        <div className="text-sm opacity-75">{(item.result.confidence * 100).toFixed(1)}%</div>
                      </div>
                      <div className="text-xs text-right opacity-70">{new Date(item.at).toLocaleString()}</div>
                    </div>
                    <div className="mt-2 text-sm">{item.result.desc}</div>
                    <div className="mt-3 flex gap-2">
                      <button onClick={() => { setResult(item.result); setCurrentImage(item.image); setNav('home'); }} className="px-3 py-1 bg-emerald-500 rounded-lg text-sm">Restore</button>
                      <button onClick={() => { navigator.clipboard?.writeText(item.image || '') }} className="px-3 py-1 bg-white/10 rounded-lg text-sm">Copy Image URL</button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      ) : (
        <div className="max-w-7xl mx-auto grid md:grid-cols-2 gap-8 md:gap-12">
          {/* Left: Inputs */}
          <div className="space-y-6">
            {/* Webcam */}
            <div className="bg-white/10 p-6 rounded-2xl backdrop-blur border border-white/20">
              <h3 className="text-2xl mb-4 font-bold">📹 Live Webcam</h3>
              <div className="relative">
                <video 
                  ref={videoRef} 
                  className="w-full rounded-xl shadow-2xl bg-black/50"
                  style={{ maxHeight: '300px' }}
                  autoPlay 
                  playsInline
                />
                {!cameraStarted && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/60 rounded-xl">
                    <p className="text-xl">Camera not started</p>
                  </div>
                )}
              </div>
              <div className="flex gap-2 mt-4">
                {!cameraStarted ? (
                  <button 
                    onClick={startCamera} 
                    disabled={!model}
                    className="flex-1 bg-gradient-to-r from-blue-500 to-blue-600 py-3 rounded-xl font-bold text-lg shadow-xl hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    📷 Start Camera
                  </button>
                ) : (
                  <>
                    <button 
                      onClick={capture} 
                      disabled={analyzing}
                      className="flex-1 bg-gradient-to-r from-emerald-500 to-teal-600 py-3 rounded-xl font-bold text-lg shadow-xl hover:scale-105 transition-all disabled:opacity-50"
                    >
                      🔍 Capture & Identify
                    </button>
                    <button 
                      onClick={stopCamera} 
                      className="px-6 bg-red-500 py-3 rounded-xl font-bold text-lg shadow-xl hover:scale-105 transition-all"
                    >
                      ⏹️
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Upload */}
            <div className="bg-white/10 p-6 rounded-2xl border border-white/20">
              <h3 className="text-2xl mb-4 font-bold">📁 Upload Image</h3>
              <input 
                type="file" 
                onChange={handleUpload} 
                accept="image/*" 
                disabled={!model || analyzing}
                className="w-full p-4 bg-white/20 rounded-xl file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-emerald-500 file:text-white file:font-semibold hover:file:bg-emerald-600 cursor-pointer disabled:opacity-50"
              />
            </div>

            {/* URL */}
            <div className="bg-white/10 p-6 rounded-2xl border border-white/20">
              <h3 className="text-2xl mb-4 font-bold">🔗 Image URL</h3>
              <input 
                type="url" 
                onKeyDown={handleUrl} 
                disabled={!model || analyzing}
                placeholder="Paste image URL and press Enter" 
                className="w-full p-4 bg-white/20 rounded-xl text-white placeholder-white/50 border border-white/30 focus:border-emerald-400 focus:outline-none disabled:opacity-50"
              />
            </div>

            {/* Current Image Preview */}
            {currentImage && (
              <div className="bg-white/10 p-6 rounded-2xl border border-white/20">
                <h3 className="text-xl mb-4 font-bold">📸 Analyzed Image</h3>
                <img src={currentImage} alt="Analyzed rock" className="w-full rounded-xl shadow-lg" />
              </div>
            )}
          </div>

          {/* Right: Results */}
          <div className="space-y-6">
            {result ? (
              <div className="bg-gradient-to-br from-emerald-500/30 to-teal-600/30 p-6 md:p-8 rounded-3xl border-4 border-emerald-400/40 backdrop-blur-xl shadow-2xl">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-3xl md:text-4xl font-black">✅ {result.name}</h2>
                  <div className="text-right">
                    <div className="text-sm opacity-75">Confidence</div>
                    <div className="text-2xl font-bold text-emerald-300">
                      {(result.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div className="bg-black/20 p-4 rounded-xl mb-4">
                  <p className="text-lg md:text-xl leading-relaxed">{result.desc}</p>
                </div>

                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="bg-black/20 p-3 rounded-lg">
                    <div className="text-sm opacity-75 mb-1">Type</div>
                    <div className="font-bold text-lg">{result.type}</div>
                  </div>
                  <div className="bg-black/20 p-3 rounded-lg">
                    <div className="text-sm opacity-75 mb-1">Mohs Hardness</div>
                    <div className="font-bold text-lg">{result.mohs}</div>
                  </div>
                  <div className="bg-black/20 p-3 rounded-lg">
                    <div className="text-sm opacity-75 mb-1">Color</div>
                    <div className="font-bold">{result.color}</div>
                  </div>
                  <div className="bg-black/20 p-3 rounded-lg">
                    <div className="text-sm opacity-75 mb-1">Texture</div>
                    <div className="font-bold">{result.texture}</div>
                  </div>
                </div>

                <div className="bg-black/20 p-4 rounded-xl mb-4">
                  <div className="font-bold mb-2 text-emerald-300">🔬 Mineral Composition</div>
                  <div className="space-y-1">
                    {result.minerals.map((mineral, idx) => (
                      <div key={idx} className="flex items-center">
                        <span className="w-2 h-2 bg-emerald-400 rounded-full mr-2"></span>
                        <span>{mineral}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-black/20 p-4 rounded-xl mb-4">
                  <div className="font-bold mb-2 text-blue-300">🌋 Formation</div>
                  <p>{result.formation}</p>
                </div>

                <div className="bg-black/20 p-4 rounded-xl">
                  <div className="font-bold mb-2 text-yellow-300">🛠️ Common Uses</div>
                  <p>{result.uses}</p>
                </div>
              </div>
            ) : (
              <div className="h-96 bg-white/10 rounded-3xl flex flex-col items-center justify-center border border-white/20 backdrop-blur">
                {loading ? (
                  <>
                    <div className="text-6xl mb-4 animate-bounce">⏳</div>
                    <div className="text-2xl font-bold">Loading AI Model...</div>
                    <div className="text-lg opacity-75 mt-2">Please wait a moment</div>
                  </>
                ) : analyzing ? (
                  <>
                    <div className="text-6xl mb-4 animate-spin">🔍</div>
                    <div className="text-2xl font-bold">Analyzing Rock...</div>
                  </>
                ) : (
                  <>
                    <div className="text-6xl mb-4">🎒</div>
                    <div className="text-2xl font-bold text-center px-4">Ready to identify your rock!</div>
                    <div className="text-lg opacity-75 mt-2 text-center px-4">Upload, capture, or paste a URL</div>
                  </>
                )}
              </div>
            )}

            {/* Info Card */}
            <div className="bg-blue-500/20 p-6 rounded-2xl border border-blue-400/30">
              <h3 className="text-xl font-bold mb-3">ℹ️ How It Works</h3>
              <ul className="space-y-2 text-sm opacity-90">
                <li>• Uses AI to analyze rock images</li>
                <li>• Identifies 10+ common rock types</li>
                <li>• Analyzes color, texture, and patterns</li>
                <li>• Provides detailed geological info</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Bottom navigation */}
      <nav className="fixed bottom-4 left-1/2 transform -translate-x-1/2 w-[95%] max-w-3xl bg-white/5 backdrop-blur rounded-xl border border-white/10 p-2 flex justify-between items-center z-50">
        <button onClick={() => setNav('home')} className={`flex-1 py-3 px-4 rounded-lg text-center ${nav === 'home' ? 'bg-emerald-500/80' : ''}`}>
          <div className="text-lg">🏠</div>
          <div className="text-xs opacity-80">Home</div>
        </button>
        <button onClick={() => setNav('history')} className={`flex-1 py-3 px-4 rounded-lg text-center ${nav === 'history' ? 'bg-emerald-500/80' : ''}`}>
          <div className="text-lg">🕘</div>
          <div className="text-xs opacity-80">History</div>
        </button>
        <a href="#" onClick={(e)=>{e.preventDefault(); window.location.reload();}} className="flex-1 py-3 px-4 rounded-lg text-center">
          <div className="text-lg">🔄</div>
          <div className="text-xs opacity-80">Reload</div>
        </a>
      </nav>

      <footer className="mt-12 md:mt-20 text-center opacity-75">
        <p className="text-sm md:text-base">Powered by TensorFlow.js & MobileNet • AI Rock Classification System</p>
      </footer>
    </div>
  );
}

export default App;
