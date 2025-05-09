from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

# Load base model & tokenizer
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"🔹 Loading base model: {base_model_id}")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16)

# Apply fine-tuned LoRA adapter
adapter_path = "toilahonganh1712/tinyllama-travelvungtau360"

print(f"🔹 Applying LoRA adapter from: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)

# Kiểm tra adapter đã load
print("🔍 Adapter configs:")
print(model.peft_config)

# Kiểm tra các tham số đang fine-tuned (có requires_grad=True)
print("🧠 Các tham số sẽ được sử dụng khi suy luận (requires_grad=True):")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  ✅ {name}")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"📦 Model loaded on: {device}")

# Flask app
app = Flask(__name__)

@app.route('/api/ask', methods=['POST'])
def chat():
    # Lấy input từ người dùng
    user_input = request.json.get("message")
    print(f"📝 Input từ user: {user_input}")

    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    # Format đúng chuẩn fine-tuning: user -> assistant
    conversation_input = f"user: {user_input}\nassistant:"
    print("📚 Đoạn hội thoại sẽ được encode:")
    print(conversation_input)

    # Tokenize đầu vào
    inputs = tokenizer(conversation_input, return_tensors="pt").to(device)

    # Generate response
    with torch.no_grad():  # Dùng với no_grad() để không tính gradient khi suy luận, tiết kiệm bộ nhớ
        output = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=300,
            do_sample=True,
            top_k=30,
            top_p=0.85,
            temperature=0.8
        )

    # Decode output
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print("📤 Output sau khi decode:")
    print(decoded_output)

    # Lấy phần sau "assistant:"
    if "assistant:" in decoded_output:
        answer = decoded_output.split("assistant:")[-1].strip()
    else:
        answer = decoded_output.strip()

    # Kiểm tra câu trả lời nếu quá ngắn
    if len(answer) < 5:
        answer = "Xin lỗi, tôi không hiểu câu hỏi của bạn hoặc không có câu trả lời phù hợp."
        print("⚠️ Trả lời quá ngắn hoặc không phù hợp.")

    print("✅ Final answer gửi về client:")
    print(answer)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    print("🚀 Flask server is running at http://localhost:5000")
    app.run(debug=True)


# --- Imports ---
# import torch
# from flask import Flask, request, jsonify
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
# from langchain_ollama.llms import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate
# # Assuming your vector store retriever setup is in 'vector.py'
# from vector import retriever # Make sure vector.py and its dependencies (FAISS, embeddings) are accessible

# # --- RAG Setup ---
# print("🔹 Initializing RAG components...")
# try:
#     # Use a more descriptive name to avoid conflict with the fine-tuned model
#     rag_llm = OllamaLLM(model="llama3.2")

#     rag_template = """
#     You are an expert in answering questions about a pizza restaurant based *only* on the provided reviews.
#     If the reviews do not contain information relevant to the question, clearly state that the information is not available in the reviews.
#     Do not make up information not present in the reviews.

#     Here are some relevant reviews:
#     {reviews}

#     Here is the question to answer: {question}

#     Answer based *only* on the reviews:
#     """
#     rag_prompt = ChatPromptTemplate.from_template(rag_template)
#     rag_chain = rag_prompt | rag_llm
#     print("✅ RAG components initialized.")
# except Exception as e:
#     print(f"⚠️ Error initializing RAG components: {e}")
#     print("Proceeding without RAG capabilities.")
#     rag_chain = None
#     retriever = None # Ensure retriever is None if setup failed

# # --- Fine-tuned Model Setup ---
# print("🔹 Initializing Fine-tuned Model components...")
# # Load base model & tokenizer
# base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# adapter_path = "toilahonganh1712/tinyllama-travelvungtau360"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# try:
#     print(f"🔹 Loading base tokenizer: {base_model_id}")
#     tokenizer = AutoTokenizer.from_pretrained(base_model_id)
#     # Add padding token if missing (important for batching and generation consistency)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         print("⚠️ Set pad_token to eos_token")

#     print(f"🔹 Loading base model: {base_model_id}")
#     # Load in lower precision if memory is a concern, but float16 is good
#     base_model = AutoModelForCausalLM.from_pretrained(
#         base_model_id,
#         torch_dtype=torch.float16, # Use float16 for faster inference and less memory
#         # device_map='auto' # Consider using device_map for automatic distribution if needed
#     )

#     # Apply fine-tuned LoRA adapter
#     print(f"🔹 Applying LoRA adapter from: {adapter_path}")
#     # Use a more descriptive name
#     finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
#     finetuned_model.eval() # Set model to evaluation mode

#     # Move the entire PEFT model to the device
#     finetuned_model.to(device)

#     print("🔍 Adapter configs:")
#     print(finetuned_model.peft_config)
#     # print("🧠 Fine-tuned parameters (requires_grad=True - Note: only for training, not inference):")
#     # for name, param in finetuned_model.named_parameters():
#     #     if param.requires_grad:
#     #         print(f"  ✅ {name}") # This just shows which ones *were* trained

#     print(f"📦 Fine-tuned Model loaded on: {device}")
#     print("✅ Fine-tuned Model components initialized.")
# except Exception as e:
#     print(f"💥 FATAL ERROR initializing fine-tuned model: {e}")
#     # Exit or handle appropriately if the core model can't load
#     exit()


# # --- Flask App ---
# app = Flask(__name__)

# @app.route('/api/ask', methods=['POST'])
# def chat():
#     user_input = request.json.get("message")
#     print(f"\n--- New Request ---")
#     print(f"📝 Input from user: {user_input}")

#     if not user_input:
#         return jsonify({"error": "Message is required"}), 400

#     final_answer = "Sorry, I encountered an issue and couldn't process your request." # Default error message
#     used_rag = False

#     # --- Step 1: Try RAG First ---
#     if rag_chain and retriever: # Check if RAG components are available
#         try:
#             print("⏳ Attempting RAG...")
#             # 1. Retrieve relevant documents
#             reviews = retriever.invoke(user_input)

#             if reviews:
#                 print(f"✅ Found {len(reviews)} relevant review(s) for RAG.")
#                 # 2. Invoke the RAG chain
#                 rag_result = rag_chain.invoke({"reviews": reviews, "question": user_input})
#                 print(f"💬 RAG Raw Output: {rag_result}")

#                 # 3. Basic Quality Check for RAG Answer
#                 #    - Is it non-empty?
#                 #    - Does it explicitly say it can't answer from reviews? (Customize this check)
#                 #    - Is it reasonably long?
#                 rag_result_stripped = rag_result.strip()
#                 negative_indicators = [
#                     "not available in the reviews",
#                     "reviews do not mention",
#                     "reviews do not contain information",
#                     "cannot answer based on the reviews",
#                     "i don't have information about that in the reviews"
#                 ]
#                 is_useful_rag_answer = (
#                     len(rag_result_stripped) > 10 and # Arbitrary minimum length
#                     not any(indicator in rag_result_stripped.lower() for indicator in negative_indicators)
#                 )

#                 if is_useful_rag_answer:
#                     final_answer = rag_result_stripped
#                     used_rag = True
#                     print("✅ Using RAG answer.")
#                 else:
#                     print("⚠️ RAG answer deemed not useful or conclusive. Falling back.")
#             else:
#                 print("ℹ️ No relevant reviews found by retriever. Skipping RAG chain.")

#         except Exception as e:
#             print(f"⚠️ Error during RAG processing: {e}. Falling back.")
#             # Fallback will happen naturally as used_rag is False

#     # --- Step 2: Fallback to Fine-tuned Model if RAG wasn't used or useful ---
#     if not used_rag:
#         print("⏳ Falling back to fine-tuned TinyLlama model...")
#         try:
#             # Format input for the fine-tuned model
#             conversation_input = f"user: {user_input}\nassistant:"
#             print("📚 Input for fine-tuned model:")
#             print(conversation_input)

#             # Tokenize
#             inputs = tokenizer(conversation_input, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device) # Added padding, truncation, max_length

#             # Generate response
#             with torch.no_grad(): # Important for inference
#                 output = finetuned_model.generate(
#                     input_ids=inputs['input_ids'],
#                     attention_mask=inputs['attention_mask'],
#                     max_new_tokens=300, # Max *new* tokens to generate
#                     do_sample=True,
#                     top_k=30,
#                     top_p=0.85,
#                     temperature=0.8,
#                     pad_token_id=tokenizer.pad_token_id # Ensure pad token is set
#                 )

#             # Decode output - only decode the generated part
#             # output[0] contains the input + output tokens
#             # inputs['input_ids'].shape[1] gives the length of the input tokens
#             generated_tokens = output[0][inputs['input_ids'].shape[1]:]
#             decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

#             print("📤 Fine-tuned Model Raw Output (decoded):")
#             print(decoded_output)

#             # Post-process: Strip whitespace
#             answer_from_finetuned = decoded_output.strip()

#             # Basic check for empty or very short answers from fine-tuned model
#             if len(answer_from_finetuned) < 5:
#                 answer_from_finetuned = "Xin lỗi, tôi không thể đưa ra câu trả lời phù hợp vào lúc này." # More specific fallback
#                 print("⚠️ Fine-tuned model answer was too short, using default fallback.")

#             final_answer = answer_from_finetuned
#             print("✅ Using Fine-tuned model answer.")

#         except Exception as e:
#             print(f"💥 Error during fine-tuned model inference: {e}")
#             # Keep the default error message set at the beginning

#     # --- Return the result ---
#     print(f"➡️ Final Answer Sent to Client (from {'RAG' if used_rag else 'Fine-tuned Model'}):")
#     print(final_answer)
#     return jsonify({"answer": final_answer})

# if __name__ == '__main__':
#     print("\n🚀 Starting Flask server...")
#     print("   - RAG Enabled: ", bool(rag_chain and retriever))
#     print("   - Fine-tuned Model Enabled: True") # Assuming it must load
#     print("   - Access API at: http://localhost:5000/api/ask (POST)")
#     # Turn off debug mode for production, keep it for development if needed
#     # Use host='0.0.0.0' to make it accessible from other devices on the network
#     app.run(host='0.0.0.0', port=5000, debug=False)