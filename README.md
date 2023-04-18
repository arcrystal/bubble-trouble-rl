# falling-circles
## Clone the repository

    $ git clone https://github.com/arcrystal/bubble-trouble-rl.git

## Setup your environment

    $ conda create -n ENV python=3.10

    $ conda activate ENV

    $ python -m pip install --upgrade pip

    $ conda install -c apple tensorflow-deps --force-reinstall -n ENV

    $ pip install -r requirements.txt

    $ pip install --no-deps keras-rl2

    $ conda env config vars set DISPLAY_WIDTH=890 FPS=52

    $ conda activate ENV

## Use Package game

    - play the game: $ python main.py

    - train a model: $ python main.py -tr -o drqn

    
