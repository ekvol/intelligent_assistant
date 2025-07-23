import React, { useState } from "react";
import axios from "axios";

const API_URL = "http://localhost:5000/api/chat";

function ChatWindow() {
  const [input, setInput] = useState("");
  const [history, setHistory] = useState([]);

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMessage = { sender: "user", text: input };
    setHistory((prev) => [...prev, userMessage]);

    try {
      const response = await axios.post(API_URL, {
        user_input: input,
        session_id: "demo-session",
      }, {
        headers: {
          Authorization: "Bearer demo-token-123",
        },
      });
      const botMessage = { sender: "bot", text: response.data.response };
      setHistory((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error(err);
      const errorMessage = {
        sender: "bot",
        text: "Ошибка при обращении к серверу.",
      };
      setHistory((prev) => [...prev, errorMessage]);
    }
    setInput("");
  };

  return (
    <div style={{ maxWidth: "600px", margin: "30px auto" }}>
      <div style={{ border: "1px solid #ccc", padding: "10px", height: "400px", overflowY: "auto", background: "#f9f9f9", borderRadius: "5px" }}>
        {history.map((msg, index) => (
          <div key={index} style={{ textAlign: msg.sender === "user" ? "right" : "left", margin: "5px 0" }}>
            <span style={{ display: "inline-block", padding: "8px 12px", borderRadius: "15px", background: msg.sender === "user" ? "#daf0ff" : "#e2ffe2" }}>
              {msg.text}
            </span>
          </div>
        ))}
      </div>
      <div style={{ marginTop: "10px", display: "flex" }}>
        <input value={input} onChange={(e) => setInput(e.target.value)} onKeyPress={(e) => e.key === "Enter" && handleSend()} placeholder="Введите ваш вопрос..." style={{ flexGrow: 1, padding: "10px", borderRadius: "5px" }} />
        <button onClick={handleSend} style={{ marginLeft: "10px", padding: "10px 20px", borderRadius: "5px", cursor: "pointer" }}>
          Отправить
        </button>
      </div>
    </div>
  );
}

export default ChatWindow;