import { useState } from "react";
import type { KeyboardEvent } from "react";
import axios from "axios";
import "./ChatUI.css";

type Message = {
  text: string;
  sender: "user" | "bot";
};

export default function ChatUI() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Add user's message
    setMessages((prev) => [...prev, { text: input, sender: "user" }]);
    setLoading(true);

    try {
      // Call backend API
      const response = await axios.post(
        "http://localhost:8000/detect-emotion",
        {
          text: input,
        }
      );

      // Get emotion from response
      const { emotion, suggestions } = response.data;

  // Add bot's message: display Gemini suggestions instead of just emotion
  setMessages((prev) => [
    ...prev,
    {
      text: `Suggestions:\n- ${suggestions.join("\n- ")}`,
      sender: "bot",
    },
  ]);
    } catch (error) {
      console.error("Error calling backend:", error);
      setMessages((prev) => [
        ...prev,
        { text: "Error: could not detect emotion", sender: "bot" },
      ]);
    } finally {
      setLoading(false);
      setInput("");
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") sendMessage();
  };

  return (
    <div className="chat-wrapper">
      <div className="chat-container">
        <div className="chat-window">
          {messages.map((msg, i) => (
            <div key={i} className={`chat-message ${msg.sender}`}>
              {msg.text}
            </div>
          ))}
          {loading && (
            <div className="chat-message bot">Detecting emotion...</div>
          )}
        </div>

        <div className="chat-input-row">
          <input
            type="text"
            placeholder="Type a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button onClick={sendMessage} disabled={loading}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
