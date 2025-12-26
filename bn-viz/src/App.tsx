import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./dashboard";
import SwapPage from "./swap_page";
import "./App.css";
import Home from "./home";

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/swap" element={<SwapPage />} />
      </Routes>
    </Router>
  );
};

export default App;
