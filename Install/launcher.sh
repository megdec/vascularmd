#!/bin/bash 

DIR=$(dirname $(readlink -f $0))

VIRT_ENV=~/virtualenvs/

if [ -d ${VIRT_ENV}"/mesh" ]
then 
    source $VIRT_ENV/mesh/bin/activate
else 
    echo "CREATE VIRTUAL ENVIRONEMENT" 
    PATH_TO_PYTHON=""
    read -p "Do you want install python 3.8 ? [y/N] : " v1
    if [ "$v1" == "y" ]
    then 
        echo "Downloading python 3.8"
        xdg-open "https://www.python.org/ftp/python/3.8.15/Python-3.8.15.tgz"
        read -p "Press enter when download is finished" v3

        echo "Select python3.8.15.tgz folder"
        FILE=`zenity --file-selection --title="select python3.8.15.tgz folder" 2>/dev/null `
        mkdir $DIR/python_installation
        cp $FILE $DIR/python_installation/Python3.8.15.tgz
        cd $DIR/python_installation
        tar xvzf Python3.8.15.tgz
        cd $DIR/python_installation/Python-3.8.15
        ./configure
        make
        make test 
        sudo make install
        cd $DIR
        PATH_TO_PYTHON="$DIR/python_installation/Python-3.8.15/python"
    else 
        read -p "Python exe is in $DIR/python_installation/Python-3.8.15/python ! PRESS ENTER " v4
        PATH_TO_PYTHON=`zenity --file-selection --title="select python3.8.15 " 2>/dev/null `        
    fi
    echo $PATH_TO_PYTHON
    if [ "$PATH_TO_PYTHON" == "" ]
    then 
        echo "ERROR PATH TO PYTHON NOT SELECTED"
        exit 2
    fi
    virtualenv $VIRT_ENV/mesh -p $PATH_TO_PYTHON
    source $VIRT_ENV/mesh/bin/activate
    pip install --upgrade pip
fi 



python --version
python module_check.py


echo "*****************************************************"
echo "**************  START APPLICATION  ******************"
echo "*****************************************************"

python main.py

