import React, { useState, useRef, useEffect } from 'react';
import './App.css';

const API_BASE_URL = 'http://0.0.0.0:8000';

function App() {
    const [targetVariable, setTargetVariable] = useState('');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [jobId, setJobId] = useState(null);
    const [jobStatus, setJobStatus] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const intervalRef = useRef(null);

    const handleTargetChange = (event) => { setTargetVariable(event.target.value); };

    const handleFileChange = (event) => {
        if (event.target.files && event.target.files[0]) {
            setSelectedFile(event.target.files[0]); } else { setSelectedFile(null);

        }
    };

    // --- Function to fetch status (will be called repeatedly) ---
    const fetchStatus = async (currentJobId) => {
        // Only fetch if we have the specific job ID we submitted
        if (!currentJobId) {
        console.log("fetchStatus called without a currentJobId, skipping.");
        stopPolling(); // Make sure polling stops if ID is lost
        return;
        }
        console.log(`Fetching status for job: ${currentJobId}`);
        setIsLoading(true); // Indicate that we are fetching

        try {
        const response = await fetch(`${API_BASE_URL}/status`); // GET request by default
        const statusResult = await response.json();

        if (!response.ok) {
            // Handle server errors or if job not found (e.g., 404)
            if (response.status === 404) {
                setError("Could not find the submitted job status on the server.");
                console.warn("Status endpoint returned 404.");
                stopPolling(); // Stop polling if job not found
            } else {
                throw new Error(statusResult.detail || `HTTP error! status: ${response.status}`);
            }
        } else {
            if (statusResult.job_id === currentJobId) {
            setJobStatus(statusResult); // Update the state with the full status object
            console.log('Fetched Status:', statusResult);

            // Check if the job is finished (completed or failed)
            if (statusResult.status === 'completed' || statusResult.status === 'failed') {
                console.log(`Job ${statusResult.status}, stopping polling.`);
                stopPolling(); // Stop polling if the job is done
            }
            } else {
                // The status endpoint returned info for a *different* job. Stop polling for the old one.
                console.warn(`Status endpoint returned data for job ${statusResult.job_id}, but we were polling for ${currentJobId}. Stopping polling.`);
                stopPolling();
            }
        }
        } catch (err) {
        console.error('Error fetching status:', err);
        setError(`Failed to fetch status: ${err.message}`);
        } finally {
        if (intervalRef.current) {
                setIsLoading(false);
        }
        }
    };

    // --- Function to start polling ---
    const startPolling = (currentJobId) => {
        stopPolling(); // Clear any existing interval before starting a new one
        console.log(`Starting polling for job: ${currentJobId}`);

        // Fetch immediately first time
        fetchStatus(currentJobId);

        // Set an interval to call fetchStatus every 5 seconds (5000ms)
        // Store the interval ID in our ref
        intervalRef.current = setInterval(() => {
        fetchStatus(currentJobId); // Pass the specific job ID
        }, 5000);
    };

    // --- Function to stop polling ---
    const stopPolling = () => {
        if (intervalRef.current) { 
        console.log("Stopping polling interval.");
        clearInterval(intervalRef.current); 
        intervalRef.current = null; 
        setIsLoading(false); 
        }
    };

    // --- Modify handleSubmit to start polling ---
    const handleSubmit = async (event) => {
        event.preventDefault();
        setError(null);
        setJobId(null);
        setJobStatus(null); // Clear previous status display
        stopPolling(); // Stop any previous polling just in case

        if (!selectedFile || !targetVariable) {
            setError('Please provide...');
              return; }

        const formData = new FormData();
        formData.append('target_variable', targetVariable);
        formData.append('file', selectedFile);

        setIsLoading(true); // Indicate loading during submission itself

        try {
        const response = await fetch(`${API_BASE_URL}/run_automl`, { method: 'POST', body: formData });
        const result = await response.json();
        if (!response.ok) { throw new Error(result.detail || `HTTP error! Status: ${response.status}`); }

        console.log('Job submitted successfully:', result);
        setJobId(result.job_id); // Set the Job ID state
        startPolling(result.job_id); // Start polling *using the received job ID*

        } catch (err) {
        console.error('Error submitting job:', err);
        setError(`Failed to submit job: ${err.message}`);
        setJobId(null);
        setIsLoading(false); // Turn off loading on submission error
        }
        // Note: setIsLoading(false) is handled by fetchStatus/stopPolling after submission succeeds
    };

    // --- useEffect for Cleanup ---
    // This effect runs only once when the component mounts (due to [])
    // The function it *returns* is the cleanup function.
    useEffect(() => {
        // This cleanup function runs when the component unmounts (is removed)
        return () => {
        console.log("App component unmounting, stopping polling.");
        stopPolling(); // Ensure polling stops if the user navigates away
        };
    }, []); // Empty dependency array means effect runs once on mount, cleanup on unmount


    // -- JSX Rendering --
    return (
        <div className="App">
          <h1>Simple AutoML Runner</h1>
    
          <form onSubmit={handleSubmit}>
             {/* ... input fields ... */}
             <div><label htmlFor="target">Target Variable:</label><input type="text" id="target" value={targetVariable} onChange={handleTargetChange} required /></div>
             <div><label htmlFor="csvfile">Upload CSV:</label><input type="file" id="csvfile" accept=".csv" onChange={handleFileChange} required /></div>
             {/* Disable button while submitting or polling */}
             <button type="submit" disabled={isLoading}>
               {isLoading ? 'Processing...' : 'Run AutoML'}
             </button>
          </form>
    
          {error && <p style={{ color: 'red' }}>Error: {error}</p>}
    
          {/* --- Updated Status Display Area --- */}
          {jobId && ( // Only show this section if a job has been submitted
            <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '10px' }}>
              <h2>Job Status</h2>
              <p><strong>Job ID:</strong> {jobId}</p>
              {isLoading && !jobStatus && <p>Submitting job...</p>} {/* Show during initial POST */}
              {isLoading && jobStatus && <p>Checking status...</p>} {/* Show during polling GET */}
    
              {/* Display details from the jobStatus object */}
              {jobStatus ? ( // Check if jobStatus state has data
                <div>
                  <p><strong>Status:</strong> {jobStatus.status}</p>
                  <p><strong>Message:</strong> {jobStatus.message}</p>
    
                  {/* Display results only if completed and results exist */}
                  {jobStatus.status === 'completed' && jobStatus.results && (
                    <div>
                      <h3>Results:</h3>
                      {/* Use <pre> for preformatted text, good for JSON */}
                      <pre style={{ padding: '10px', overflowX: 'auto' }}>
                        {/* Nicely format the metrics object as a JSON string */}
                        {JSON.stringify(jobStatus.results.metrics, null, 2)}
                      </pre>
                      {/* Display file paths (these are relative to your backend project root) */}
                      <p><strong>Model File:</strong> {jobStatus.results.model_file}</p>
                      <p><strong>Log File:</strong> {jobStatus.results.log_file}</p>
                      {/* In a real app, you might make these downloadable links */}
                    </div>
                  )}
                   {/* Display message if failed */}
                   {jobStatus.status === 'failed' && (
                     <p style={{ color: 'orange' }}>Job failed. Check backend logs for details.</p>
                   )}
                </div>
              ) : (
                 // Show message if we have a job ID but no status data yet (e.g., before first poll)
                !isLoading && <p>Waiting for first status update...</p>
              )}
            </div>
          )}
        </div>
      );
}

export default App;