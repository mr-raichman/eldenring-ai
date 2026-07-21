Elden Ring AI Notes by Teo Raichman :)

I've added the command "elden" to ~/.bashrc in order to alias cd ~/elden-ai and source/venv/bin/activate
In order to exit the virtual enviroment, use the command "deactivate"

The pixels I've mesured for the health bars are:
Player: (154, 47); (367, 60)
Boss: (466, 866); (1463, 878)


Create shared terminal:
tmux new -s shared

Access that in laptop:
Alias = remote_terminal
ssh pingu@192.168.1.43 -L 6006:localhost:6006 -t tmux attach -t shared


Copy terminal:
tmux capture-pane -p -S - > output.txt


Flask count: +0x388
Flask binary: +0x3B0
FLASK_STATIC_PTR = 0x6FFFFA15B880


Direction for area: +0x19A
68 = before margit
4 = Margit arena
0 = Before respawning


Action durations:
_ACTION_DURATIONS_REF = {
    "No Action":    0.10,
    "Dodge":        0.75,
    "Light Attack": 0.55,
    "Heavy Attack": 0.80,
    "Heal":         1.30,
    "Jump":         0.55,
    "Parry":        0.70,
    "Lock Camera":  0.05,
}


Start Elden Ring (workspace 4):
steam steam://rungameid/1245620

Input keys:
xdotool key space          # Spacebar
xdotool key Escape         # Escape
xdotool key ctrl+s         # Ctrl+S
xdotool key F1             # Function key
xdotool key a              # Letter 'a'

Change workspace:
hyprctl dispatch workspace X

Take a screenshot and save it:
grim ~/eldenring-ai/orientation.jpg

Copy to laptop (command in laptop):
scp pingu@192.168.1.43:~/eldenring-ai/orientation.jpg ~/Desktop/
