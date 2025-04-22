import React, { useState } from 'react';
import './App.css';

function App() {
    const [targetVariable, setTargetVariable] = useState('');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);

    // -- Event Handlers --
    const handleTargetChange = (event: React.ChangeEvent<HTMLInputElement>) =>  {
        setTargetVariable(event.target.value)
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            setSelectedFile(event.target.files[0]);
        } else {
            setSelectedFile(null);
        }
    };

    // -- JSX Rendering --
    return (
        <div className="App">
        <h1>AutoML Runner</h1>

        <form>
            <div>
            <label htmlFor="target">Target Variable:</label>
            <input
                type="text"
                id="target"
                value={targetVariable}
                onChange={handleTargetChange}
            />
            </div>
            <div>
            <label htmlFor="csvfile">Upload CSV:</label>
            {/* File input */}
            <input
                type="file"
                id="csvfile"
                accept=".csv"
                onChange={handleFileChange}
            />
            </div>
            {/* Submit button */}
            <button type="submit">Run AutoML</button>
        </form>

        {/* Display the current state values (for testing/debugging) */}
        <div style={{ marginTop: '20px' }}>
            <h3>Current State:</h3>
            <p>Target Variable: {targetVariable}</p>
            <p>Selected File: {selectedFile ? selectedFile.name : 'None'}</p>
        </div>
        </div>
    );
};

export default App;