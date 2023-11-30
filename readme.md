
# StyleBot-DokusoAI

## Overview
StyleBot-DokusoAI is an advanced chatbot designed for fashion e-commerce. It leverages OpenAI's language models to provide an interactive and intelligent conversational experience. Integrated with DokusoAPI, it assists users in finding fashion items, offering personalized recommendations, and answering queries related to fashion products.

## Features
- **Natural Language Processing:** Utilizes OpenAI's models for understanding and responding to user queries.
- **API Integration:** Interacts with Dokuso's fashion e-commerce backend to fetch and display product information.
- **Dynamic Conversational Flow:** Managed by LangChain, enabling coherent and contextually aware dialogues.
- **Data Validation:** Ensures robust data handling using Pydantic.
- **Web-based User Interface:** Interactive chat interface powered by Panel.

## Installation

Clone the repository
```
git clone https://github.com/Dokuso-App/StyleBot-DokusoAI.git
```

Install dependencies
```
pip install -r requirements.txt
```

## Usage
```
panel serve main.py
```