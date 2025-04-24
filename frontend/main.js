const models = {
    "OpenAI": ["gpt-4o", "gpt-4o-mini"],
    "Groq": ["llama3-70b-8192", "mixtral-8x7b"]
  };
  
  function updateModelDropdown() {
    const provider = document.getElementById("provider").value;
    const modelSelect = document.getElementById("model");
  
    modelSelect.innerHTML = "";
  
    models[provider].forEach(model => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      modelSelect.appendChild(option);
    });
  }
  
  // Initialize default options
  updateModelDropdown();
  
  async function submitQuery() {
    const provider = document.getElementById("provider").value;
    const model = document.getElementById("model").value;
    const query = document.getElementById("query").value;
  
    const payload = {
      model_name: model,
      model_provider: provider,
      system_prompt: "You are a helpful assistant",
      messages: [query],
      allow_search: false
    };
  
    const responseBox = document.getElementById("response");
    responseBox.textContent = "Loading...";
  
    try {
      const response = await fetch("http://127.0.0.1:9999/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });
  
      if (!response.ok) throw new Error("Failed to get response");
  
      const result = await response.json();
      responseBox.textContent = result.response || JSON.stringify(result, null, 2);
    } catch (err) {
      responseBox.textContent = `Error: ${err.message}`;
    }
  }
  