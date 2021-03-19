from flask import Flask,render_template,request
from transformers import T5ForConditionalGeneration, T5Tokenizer
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi as ytt
# define a variable to hold you app
app = Flask(__name__)

# initialize the model architecture and weights
model = T5ForConditionalGeneration.from_pretrained("t5-small")
# initialize the model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
# define your resource endpoints
@app.route('/')
def index_page():
    return render_template('index.html')
#extracts id from url
def extract_video_id(url:str):
    # Examples:
    # - http://youtu.be/SA2iWivDJiE
    # - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    # - http://www.youtube.com/embed/SA2iWivDJiE
    # - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    query = urlparse(url)
    if query.hostname == 'youtu.be': return query.path[1:]
    if query.hostname in {'www.youtube.com', 'youtube.com'}:
        if query.path == '/watch': return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/embed/': return query.path.split('/')[2]
        if query.path[:3] == '/v/': return query.path.split('/')[2]
    # fail?
    return None
#text summarizer
def summarizer(script):
    # encode the text into tensor of integers using the appropriate tokenizer
    input_ids = tokenizer("summarize: " + script, return_tensors="pt", max_length=512, truncation=True).input_ids
    # generate the summarization output
    outputs = model.generate(
        input_ids, 
        max_length=150, 
        min_length=40, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True)

    summary_text = tokenizer.decode(outputs[0])
    return(summary_text)


@app.route('/summarize',methods=['GET','POST'])
def video_transcript():
    if request.method == 'POST':
        url = request.form['youtube_url']
        video_id = extract_video_id(url)
        data = ytt.get_transcript(video_id,languages=['de', 'en'])
        
        scripts = []
        for text in data:
            for key,value in text.items():
                if(key=='text'):
                    scripts.append(value)
        transcript = " ".join(scripts)
        summary = summarizer(transcript)
        return(summary)
    else:
        return "ERROR"

    
    
# server the app when this file is run
if __name__ == '__main__':
    app.run()