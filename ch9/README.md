# Chapter 9: Exploring the Learning Environments Landscape

## [Roboschool](https://github.com/openai/roboschool)

To setup Roboschool in the `rl_gym_book` conda environment, follow the steps below:

1. Activate the `rl_gym_book` conda environment: `source activate rl_gym_book`
2. Navigate to this (`ch9`) folder and run the `setup_roboschool.sh` script: 
   1. `cd ch9`
   2. `chmod a+x ./setup_roboschool.sh`
   3. `./setup_roboschool.sh`
3. Follow the output from the script to make sure the installation is successful.
You can run a demo using the following command:

   `(rl_gym_book) praveen@ubuntu:~/HOIAWOG/ch9$ python ~/software/roboschool/agent_zoo/
demo_race2.py`
  
**Note**:
If you get an error like below when you launch the demo script or when you try to use a Roboschool environment:

```bash
QGLShaderProgram: could not create shader program
bool QGLShaderPrivate::create(): Could not create shader of type 2.
python: render-simple.cpp:250: void SimpleRender::Context::initGL(): Assertion `r0' failed.
```

Then, add the following line at the top of the demo script or to your script which are are trying to run:
`from OpenGL import GLU`