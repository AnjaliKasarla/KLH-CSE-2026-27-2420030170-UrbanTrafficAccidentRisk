import { BrowserRouter, Route, Routes } from "react-router-dom";

import { AppShell } from "./layouts/AppShell";

import Overview from "./pages/Overview";
import RiskAssessment from "./pages/RiskAssessment";
import RiskIntelligence from "./pages/RiskIntelligence";
import Analytics from "./pages/Analytics";
import Explainability from "./pages/Explainability";
import SafetyIntelligence from "./pages/SafetyIntelligence";

function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<Overview />} />

          <Route
            path="/risk-assessment"
            element={<RiskAssessment />}
          />

          <Route
            path="/risk-intelligence"
            element={<RiskIntelligence />}
          />

          <Route
            path="/analytics"
            element={<Analytics />}
          />

          <Route
            path="/explainability"
            element={<Explainability />}
          />

          <Route
            path="/safety-intelligence"
            element={<SafetyIntelligence />}
          />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}

export default App;