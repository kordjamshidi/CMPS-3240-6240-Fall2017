This assignment is a part of your class project assignment. I would like to ask people who do not have a GitHub user, make one and join the class Github repository for your projects. 

Instructions: 
1) Make a Github account in here: https://github.com/ 
2) Log in to your account
3) Go to https://github.com/kordjamshidi/CMPS-4720-6720 
4) On the top right corner of this project's Github page, you see a Fork button, press Fork!

Now you have your own repository which is a kind of copy of my project and you are able to work from there. 
To start your work on your local machine do the following steps:
1) Go to a command prompt terminal on your machine.
2) Go to the directory that you want to have your project
3) Go to your Github web page and look for the green button of “Clone or download” click on it, you will see a small window with the git address of your GitHub page copy this address!
4) Go back to the terminal and type: `git clone paste-your-copied-address`

Now you have a local copy of your repository.
1) type: `cd CMPS-4720-6720` 
2) type: `git remote add upstream https://github.com/kordjamshidi/CMPS-3240-6240-Fall2017.git`
3) type: `git remote -v`

You should be able to see the following lines after this if everything has been done correctly: 
```
origin https://github.com/YourRepoName/CMPS-4720-6720.git (fetch)
origin https://github.com/YourRepoName/CMPS-4720-6720.git (push)
upstream https://github.com/kordjamshidi/CMPS-4720-6720.git (Links to an external site.) (fetch)
upstream https://github.com/kordjamshidi/CMPS-4720-6720.git (Links to an external site.) (push)
```
Please submit the URL of your Fork to canvas for this exercise. 
