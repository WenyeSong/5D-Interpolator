"use client";

import { useState } from "react"; 
import "./styles.css";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts";

export default function Home() {
  const [uploadMsg, setUploadMsg] = useState("");
  const [trainMsg, setTrainMsg] = useState("");
  const [predictMsg, setPredictMsg] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [inputs, setInputs] = useState(["0", "0", "0", "0", "0"]);
  const [layers, setLayers] = useState("64,32,16");  // default value
  const [lr, setLR] = useState("0.01");
  const [epochs, setEpochs] = useState("200");
  const [uploadData, setUploadData] = useState<any>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainStatus, setTrainStatus] = useState("");
  const [earlyStopInfo, setEarlyStopInfo] = useState<any>(null);
  const [benchmarkResults, setBenchmarkResults] = useState<any>(null);
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [benchmarkStatus, setBenchmarkStatus] = useState("");

  // Upload dataset
  const upload = async () => {
    if (!file) {
      setUploadMsg("Please choose a .pkl file first.");
      return;
    }

    // upload form
    const formData = new FormData();
    formData.append("file", file); 
    
    //send request
    const res = await fetch("http://127.0.0.1:8000/upload", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errorText = await res.text();
      let errorDetail = `HTTP Error ${res.status}: ${res.statusText}`;
      try {
        const errorJson = JSON.parse(errorText);
        errorDetail = `Error (${res.status}): ${errorJson.detail || errorJson.message || errorText}`;
      } catch {
        errorDetail = `Error (${res.status}): ${errorText}`;
      }
      setUploadMsg(errorDetail);
      return;
    }

    const data = await res.json();   
  
    if (!res.ok) {
      setUploadMsg(data.detail || "Upload failed.");
      return;
    }
    setUploadData(data);    
    setUploadMsg("Upload success!");
  };


  // Train model
  const train = async () => {
    setIsTraining(true);
    setTrainStatus("Training...");
    setTrainMsg("");
    // parameters
    const payload = {
      layers: layers.split(",").map((n) => parseInt(n.trim())), // string to int
      lr: parseFloat(lr),
      epochs: parseInt(epochs),
    };

    const res = await fetch("http://127.0.0.1:8000/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    
    if (data.early_stopped) {
      setEarlyStopInfo({
        stopped_epoch: data.stopped_epoch,
        best_epoch: data.best_epoch,
        best_val_loss: data.best_val_loss,
        total_epochs: data.total_epochs
      });
      setTrainStatus(`Training stopped early at epoch ${data.stopped_epoch} (best at epoch ${data.best_epoch})`);
    } else {
      setEarlyStopInfo(null);
      setTrainStatus("Training completed (all epochs finished)");
    }
    
    setTrainMsg(data.logs.join("\n")); 

    setIsTraining(false);
  };


  // Predict
  const predict = async () => {
    const X = inputs.map((x) => {
      const num = parseFloat(x);
      if (isNaN(num)) {
        return 0;
      }
      return Math.max(-1, Math.min(1, num));
    });

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ X }),
      });

      if (!res.ok) {
        const errorData = await res.json();
        setPredictMsg(`Error: ${errorData.detail || "Prediction failed"}`);
        return;
      }

      const data = await res.json();
      setPredictMsg(`Prediction: ${data.y_pred.toFixed(4)}`);
    } catch (error: any) {
      setPredictMsg(`Error: ${error.message || "Failed to connect to server"}`);
    }
  };

  // Benchmark
  const runBenchmark = async () => {
    setIsBenchmarking(true);
    setBenchmarkStatus("Running benchmark... This may take a few minutes.");
    setBenchmarkResults(null);

    const payload = {
      layers: layers.split(",").map((n) => parseInt(n.trim())),
      lr: parseFloat(lr),
      epochs: parseInt(epochs),
      dataset_sizes: [1000, 5000, 10000]
    };

    try {
      const res = await fetch("http://127.0.0.1:8000/benchmark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const responseText = await res.text();
      
      if (!res.ok) {
        let errorDetail = `HTTP Error ${res.status}: ${res.statusText}`;
        try {
          const errorData = JSON.parse(responseText);
          errorDetail = errorData.detail || errorData.message || errorDetail;
        } catch {
          if (responseText) {
            errorDetail = responseText;
          }
        }
        setBenchmarkStatus(`Error (${res.status}): ${errorDetail}`);
        setIsBenchmarking(false);
        return;
      }

      try {
        const data = JSON.parse(responseText);
        if (data.results && Array.isArray(data.results)) {
          setBenchmarkResults(data.results);
          setBenchmarkStatus("Benchmark completed!");
        } else {
          setBenchmarkStatus(`Error: Invalid response format. Response: ${responseText.substring(0, 100)}`);
        }
      } catch (parseError) {
        setBenchmarkStatus(`Error: Failed to parse response. Response: ${responseText.substring(0, 100)}`);
      }
    } catch (error: any) {
      const errorMsg = error.message || "Unknown error";
      if (errorMsg.includes("Failed to fetch") || errorMsg.includes("NetworkError") || errorMsg.includes("fetch")) {
        setBenchmarkStatus("Error: Cannot connect to backend. Please check: 1) Backend is running on http://127.0.0.1:8000, 2) /benchmark endpoint exists, 3) No CORS issues.");
      } else {
        setBenchmarkStatus(`Error: ${errorMsg}`);
      }
    } finally {
      setIsBenchmarking(false);
    }
  };


  // UI Section
  return (
    <div className="page-container">
      <h1 className="page-title">5D Interpolator</h1>
      {/* --- Project Introduction --- */}
      <p className="intro">
      Welcome to a simple 5D interpolator! This interpolator lets you upload 5D data, train a customizable neural network, and generate predictions easily.
      </p>

      {/* Upload Section */}
      <div className="card">
        <h2>1. Upload Dataset</h2>

        <input
          type="file"
          accept=".pkl"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="input-box"
        />
        <br />

        <button onClick={upload} className="main-button">
          Upload
        </button>

        {uploadData && (
  <div className="upload-summary">
  <h3>Dataset Uploaded Successfully!</h3>

  <h4>Dataset Information</h4>
  <ul>
    <li><strong>{uploadData.data_points}</strong> data points</li>
    <li><strong>{uploadData.number_features}</strong> input features</li>
    <li>
      Target range: 
      [{uploadData.target_min.toFixed(3)}, {uploadData.target_max.toFixed(3)}]
    </li>
  </ul>

  <h4>Preview (first 5 rows):</h4>
  <table className="preview-table">
  <thead>
    <tr>
      {Array.from({ length: uploadData.number_features }, (_, i) => (
        <th key={i}>X{i + 1}</th>
      ))}
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    {uploadData.preview?.map((row: number[], idx: number) => (
      <tr key={idx}>
        {row.map((v, j) => (
          <td
            key={j}
            style={{
              fontWeight: j === row.length - 1 ? "bold" : "normal",
              color: j === row.length - 1 ? "#1e40af" : "inherit",
            }}
          >
            {v.toFixed(3)}
          </td>
        ))}
      </tr>
    ))}
  </tbody>
</table>


  </div>
)}
      </div>


      {/* Train Section */}
      <div className="card">
        <h2>2. Train Model</h2>

        <div>
          Layers:{" "}
          <input
            value={layers}
            onChange={(e) => setLayers(e.target.value)}
            className="input-box"
          />
        </div>

        <div>
          LR:{" "}
          <input
            value={lr}
            onChange={(e) => setLR(e.target.value)}
            className="input-box"
          />
        </div>

        <div>
          Epochs:{" "}
          <input
            value={epochs}
            onChange={(e) => setEpochs(e.target.value)}
            className="input-box"
          />
        </div>

        <button onClick={train} className="main-button" disabled={isTraining}>
            {isTraining ? "Training..." : "Train"}
      </button>
      {(trainStatus || isTraining) && (
  <div style={{ marginTop: "10px" }}>
    <p style={{ 
      fontWeight: "bold", 
      color: earlyStopInfo ? "#f59e0b" : "#059669",
      fontSize: "16px",
      padding: "10px",
      backgroundColor: earlyStopInfo ? "#fef3c7" : "#d1fae5",
      borderRadius: "6px",
      border: `2px solid ${earlyStopInfo ? "#f59e0b" : "#059669"}`
    }}>
      {trainStatus || "Training..."}
    </p>

    {earlyStopInfo && (
      <div style={{
        marginTop: "10px",
        padding: "12px",
        backgroundColor: "#fff7ed",
        borderRadius: "6px",
        border: "1px solid #fed7aa"
      }}>
        <p style={{ margin: "4px 0", fontSize: "14px" }}>
          <strong>Early Stop Details:</strong>
        </p>
        <ul style={{ margin: "8px 0", paddingLeft: "20px", fontSize: "13px" }}>
          <li>Stopped at epoch: <strong>{earlyStopInfo.stopped_epoch}</strong> / {earlyStopInfo.total_epochs}</li>
          <li>Best epoch: <strong>{earlyStopInfo.best_epoch}</strong></li>
          <li>Best validation loss: <strong>{earlyStopInfo.best_val_loss?.toFixed(6)}</strong></li>
        </ul>
      </div>
    )}
  </div>
)}

{trainMsg && (
        <pre className="code-block" style={{ 
          maxHeight: "400px", 
          overflowY: "auto",
          position: "relative"
        }}>
          {trainMsg.split('\n').map((line, idx) => {
            const isEarlyStop = line.includes('Early stopping');
            return (
              <div
                key={idx}
                style={{
                  backgroundColor: isEarlyStop ? '#fef3c7' : 'transparent',
                  padding: isEarlyStop ? '4px 8px' : '0',
                  margin: isEarlyStop ? '4px 0' : '0',
                  borderRadius: isEarlyStop ? '4px' : '0',
                  fontWeight: isEarlyStop ? 'bold' : 'normal',
                  color: isEarlyStop ? '#92400e' : 'inherit',
                  borderLeft: isEarlyStop ? '3px solid #f59e0b' : 'none',
                  paddingLeft: isEarlyStop ? '12px' : '0'
                }}
              >
                {line}
              </div>
            );
          })}
        </pre>
        )}
      </div>


      {/* Predict Section */}
      <div className="card">
        <h2>3. Predict</h2>
        <p style={{ marginBottom: "15px", color: "#666" }}>
          Adjust the sliders or enter values directly (range: -1.0 to 1.0) for each of the 5 input features.
          Make sure you have trained a model first.
        </p>
        {inputs.map((v, i) => (
          <div key={i} className="slider-row">
            <label className="slider-label">X{i + 1}</label>

            <input
              type="range"
              min="-1"
              max="1"
              step="0.01"
              value={v || "0"}
              onChange={(e) => {
                const copy = [...inputs];
                copy[i] = e.target.value;
                setInputs(copy);
              }}
              className="slider"
            />

            <input
              type="number"
              value={v || "0"}
              className="slider-value"
              min="-1"
              max="1"
              step="0.01"
              onChange={(e) => {
                const value = e.target.value;
                const numValue = parseFloat(value);
                if (!isNaN(numValue) && numValue >= -1 && numValue <= 1) {
                  const copy = [...inputs];
                  copy[i] = value;
                  setInputs(copy);
                } else if (value === "" || value === "-") {
                  const copy = [...inputs];
                  copy[i] = value;
                  setInputs(copy);
                }
              }}
            />
          </div>
        ))}

<button onClick={predict} className="main-button">
  Predict
</button>

{predictMsg && (
  <div className="prediction-box">
    {predictMsg}
  </div>
)}
</div>

      {/* Benchmark Section */}
      <div className="card">
        <h2>4. Performance Benchmark</h2>
        <p style={{ marginBottom: "15px", color: "#666" }}>
          Run performance analysis across different dataset sizes (1K, 5K, 10K samples).
          This will measure training time, memory usage, and accuracy metrics.
        </p>

        <button 
          onClick={runBenchmark} 
          className="main-button" 
          disabled={isBenchmarking}
        >
          {isBenchmarking ? "Running Benchmark..." : "Run Benchmark"}
        </button>

        {benchmarkStatus && (
          <p style={{ 
            fontWeight: "bold", 
            marginTop: "10px",
            color: isBenchmarking ? "#f59e0b" : "#059669"
          }}>
            {benchmarkStatus}
          </p>
        )}

        {benchmarkResults && benchmarkResults.length > 0 && (
          <div style={{ marginTop: "30px" }}>
            <h3 style={{ marginBottom: "20px" }}>Benchmark Results</h3>
            
            {/* Summary Table */}
            <div style={{ marginBottom: "30px", overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ backgroundColor: "#f4f4f4" }}>
                    <th style={{ padding: "10px", border: "1px solid #ddd", textAlign: "left" }}>Dataset Size</th>
                    <th style={{ padding: "10px", border: "1px solid #ddd", textAlign: "left" }}>Time (s)</th>
                    <th style={{ padding: "10px", border: "1px solid #ddd", textAlign: "left" }}>Train Mem (MB)</th>
                    <th style={{ padding: "10px", border: "1px solid #ddd", textAlign: "left" }}>Pred Mem (MB)</th>
                    <th style={{ padding: "10px", border: "1px solid #ddd", textAlign: "left" }}>MSE</th>
                    <th style={{ padding: "10px", border: "1px solid #ddd", textAlign: "left" }}>R²</th>
                  </tr>
                </thead>
                <tbody>
                  {benchmarkResults.map((r: any, idx: number) => (
                    <tr key={idx}>
                      <td style={{ padding: "10px", border: "1px solid #ddd" }}>{r.dataset_size}</td>
                      <td style={{ padding: "10px", border: "1px solid #ddd" }}>{r.training_time_s.toFixed(2)}</td>
                      <td style={{ padding: "10px", border: "1px solid #ddd" }}>{r.training_peak_memory_mb.toFixed(2)}</td>
                      <td style={{ padding: "10px", border: "1px solid #ddd" }}>{r.prediction_peak_memory_mb.toFixed(2)}</td>
                      <td style={{ padding: "10px", border: "1px solid #ddd" }}>{r.mse.toFixed(6)}</td>
                      <td style={{ padding: "10px", border: "1px solid #ddd" }}>{r.r2.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Charts */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", marginTop: "30px" }}>
              {/* Training Time Chart */}
              <div>
                <h4 style={{ marginBottom: "10px" }}>Training Time vs Dataset Size</h4>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={benchmarkResults}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="dataset_size" 
                      label={{ value: "Dataset Size", position: "insideBottom", offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: "Time (seconds)", angle: -90, position: "insideLeft" }}
                    />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="training_time_s" 
                      stroke="#2E86AB" 
                      strokeWidth={2}
                      dot={{ r: 5 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Memory Usage Chart */}
              <div>
                <h4 style={{ marginBottom: "10px" }}>Memory Usage vs Dataset Size</h4>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={benchmarkResults}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="dataset_size" 
                      label={{ value: "Dataset Size", position: "insideBottom", offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: "Memory (MB)", angle: -90, position: "insideLeft" }}
                    />
                    <Tooltip />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="training_peak_memory_mb" 
                      stroke="#A23B72" 
                      strokeWidth={2}
                      dot={{ r: 5 }}
                      name="Training Memory"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="prediction_peak_memory_mb" 
                      stroke="#F18F01" 
                      strokeWidth={2}
                      dot={{ r: 5 }}
                      name="Prediction Memory"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* MSE Chart */}
              <div>
                <h4 style={{ marginBottom: "10px" }}>MSE vs Dataset Size</h4>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={benchmarkResults}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="dataset_size" 
                      label={{ value: "Dataset Size", position: "insideBottom", offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: "MSE", angle: -90, position: "insideLeft" }}
                    />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="mse" 
                      stroke="#C73E1D" 
                      strokeWidth={2}
                      dot={{ r: 5 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* R² Chart */}
              <div>
                <h4 style={{ marginBottom: "10px" }}>R² Score vs Dataset Size</h4>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={benchmarkResults}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="dataset_size" 
                      label={{ value: "Dataset Size", position: "insideBottom", offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: "R² Score", angle: -90, position: "insideLeft" }}
                    />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="r2" 
                      stroke="#06A77D" 
                      strokeWidth={2}
                      dot={{ r: 5 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* --- Footer --- */}
      <footer className="footer">
        Developed by <strong>Wenye Song</strong> © 2025 ws452@cam.ac.uk
    </footer>

    </div>
  );
}
