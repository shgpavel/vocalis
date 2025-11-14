# vocalis
Yet another voice personal assistant

> **⚠️** PURE AI SLOP

in early development


## running
At current state it does run STT via local
openai whisper model

* get yourself a venv lt:
```
python -m venv venv
source venv/bin/activate
```

* install all the deps:\
(no requirements.txt bc I tested only Linux+Intel Xe GPU)
```
pip install ...

# you def need openai-whisper, sounddevice, torch, silero-vad
```

* run it:
```
cd ..
python -m vocalis
```
