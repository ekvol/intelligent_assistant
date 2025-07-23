import React from "react";
import ChatWindow from "./components/ChatWindow";

function App() {
  return (
    <div className="App">
      <h1 style={{ textAlign: "center" }}>
        Интеллектуальный помощник абитуриентов
      </h1>
      <ChatWindow />
    </div>
  );
}

export default App;