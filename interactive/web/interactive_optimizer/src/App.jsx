import './App.css'
import RunDashboard from './components/ExampleControl.jsx'

// const host = "localhost:9876"; // Replace with your FastAPI server host and port

function App() {
  return (
    <>
    <h1 className="text-3xl font-bold underline">
      Interactive Optimizer
    </h1>
      <RunDashboard host={"ws://localhost:9876/ws/client"}/>
    </>
  )
}

export default App
