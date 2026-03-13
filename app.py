from fastapi import FastAPI
from qnmodel import ask_question
from fastapi.responses import HTMLResponse
from fastapi import Body
from fastapi import Depends




app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Policy AI Assistant</title>

<style>

body{
    font-family: Arial, sans-serif;
    background:#eef2f7;
    display:flex;
    justify-content:center;
    align-items:center;
    height:100vh;
    margin:0;
}

/* Main card */

.container{
    width:750px;
    height:600px;
    background:white;
    border-radius:12px;
    box-shadow:0px 10px 25px rgba(0,0,0,0.1);
    display:flex;
    flex-direction:column;
}

/* Header */

.header{
    background:#007bff;
    color:white;
    padding:15px;
    text-align:center;
    font-size:20px;
    font-weight:bold;
    border-top-left-radius:12px;
    border-top-right-radius:12px;
}

/* Chat area */

.chatbox{
    flex:1;
    padding:15px;
    overflow-y:auto;
    background:#f9fbfd;
}

/* Messages */

.message{
    margin:10px 0;
    padding:10px 14px;
    border-radius:10px;
    max-width:70%;
    line-height:1.4;
}

.user{
    background:#007bff;
    color:white;
    margin-left:auto;
}

.bot{
    background:#e5e7eb;
    color:#333;
}

/* Input area */

.input-area{
    display:flex;
    padding:12px;
    border-top:1px solid #ddd;
    background:white;
}

input{
    flex:1;
    padding:10px;
    border-radius:6px;
    border:1px solid #ccc;
    outline:none;
}

button{
    padding:10px 20px;
    margin-left:8px;
    background:#007bff;
    color:white;
    border:none;
    border-radius:6px;
    cursor:pointer;
}

button:hover{
    background:#0056b3;
}

</style>
</head>

<body>

<div class="container">

<div class="header">
Policy AI Assistant
</div>

<div id="chatbox" class="chatbox"></div>

<div class="input-area">
<input id="question" placeholder="Ask a question..." />
<button onclick="ask()">Send</button>
</div>

</div>

<script>
window.onload = function(){

let chatbox = document.getElementById("chatbox");

chatbox.innerHTML += `
<div class="message bot">
Hello 👋 I am the KADIT Policy AI Assistant. I can help answer your questions about the company policy document. <br>
Please ask your question below.
</div>
`;

}

async function ask(){

let input = document.getElementById("question");
let question = input.value;

if(question.trim() === "") return;

let chatbox = document.getElementById("chatbox");

chatbox.innerHTML += `<div class="message user">${question}</div>`;
chatbox.innerHTML += `<div class="message bot" id="loading">Thinking...</div>`;

chatbox.scrollTop = chatbox.scrollHeight;

input.value="";

let response = await fetch("/ask",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({question:question})
});

let data = await response.json();

document.getElementById("loading").remove();

chatbox.innerHTML += `<div class="message bot">${data.answer}</div>`;

chatbox.scrollTop = chatbox.scrollHeight;

}

document.getElementById("question").addEventListener("keypress",function(e){
if(e.key === "Enter"){
ask();
}
});

</script>

</body>
</html>
"""

@app.post("/ask")
def ask(question: dict = Body(...)):
    q = question["question"]
    answer = ask_question(q)


   
    return {"answer": answer}