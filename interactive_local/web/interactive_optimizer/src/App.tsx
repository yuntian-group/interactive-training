import NavigationBar from './components/navbar/navbar';
import ControlBar from './components/controlBar/controlBar';
import './App.css'

function App() {
  return (
    <div className="App h-screen w-screen bg-gray-100">
      <header className="App-header flex-shrink-0 h-1/12">
        <NavigationBar className="h-full" />
      </header>
      <div className="flex flex-row flex-1 overflow-auto h-11/12 w-full">
        <ControlBar className="h-full w-1/3"/>
      </div>
    </div>
  )
}

export default App
