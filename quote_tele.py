import pandas as pd
import telebot
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset

TOKEN = '5985198127:AAFHP5fSqquphKkIeFqEUmemhtWBKqnaXBg'

bot = telebot.TeleBot(TOKEN)

def process_csv(df):
    qa_pairs = []
    for index, row in df.iterrows():
        quote = row['quote']
        author = row['author']
        qa_pairs.append(f"Quote: {quote}\nAuthor: {author}\n")
    return qa_pairs

def load_dataset(df, tokenizer):
    qa_pairs = process_csv(df)
    tokenized_dataset = tokenizer(qa_pairs, truncation=True,
                                  padding='max_length', max_length=128,
                                  return_tensors="pt")
    dataset = Dataset.from_dict(tokenized_dataset)
    return dataset

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Hello! Please provide a quote, and I will try to guess the author.")

@bot.message_handler(func=lambda message: True)
def guess_author(message):
    quote = message.text

    # Load the fine-tuned model and tokenizer
    model_name = "fine_tuned_QuotesandAuthors_gpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    answer = ask_question(quote, model, tokenizer)

    reply = f"Quote: {quote}\nAuthor: {answer}"
    bot.reply_to(message, reply)

def ask_question(quote, model, tokenizer, max_length=128, num_return_sequences=1):
    prompt = f"Quote: {quote}\nAuthor:"
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=3,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        early_stopping=True,
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.replace(prompt, "").strip()

    # Truncate the answer after the first newline character
    answer = answer.split("\n")[0]

    return answer

if __name__ == "__main__":
    # Load and preprocess the dataset
    quotesto = pd.read_csv("quote.csv", encoding="latin-1")
    valid = pd.read_csv("valid.csv", encoding="latin-1")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    train_dataset = load_dataset(quotesto, tokenizer)
    valid_dataset = load_dataset(valid, tokenizer)

    # Configure and train the model using the Trainer class
    training_args = TrainingArguments(
        output_dir="output",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_steps=100,
        save_steps=100,
        warmup_steps=0,
        logging_dir="logs",
        evaluation_strategy="steps",
        save_total_limit=3,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    trainer.train()
    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_QuotesandAuthors_gpt2")

    bot.polling()
