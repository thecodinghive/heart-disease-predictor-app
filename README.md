TODO: This readme was copied from the digit identifier. Update with new instructions specific to heart disease model.

TODO: Livereload would be nice too (though optional...)
    https://livereload.readthedocs.io/en/latest/

Run using:

    python3 app.py

Setup:

* TODO: Setup dependencies via conda.

    # Initial env creation
    conda env create -f environment.yml
    # USe env
    conda activate flask-heart-disease
    # update env (e.g. after updating deps?)
    conda env update --file environment.yml --prune


# Running on Heroku

After installing new packages, make sure to update `requirements.txt`:

    conda list -e > requirements.txt

The `Procfile` contains the command needed to run the app:

    web: gunicorn app:app --log-file -

Install Heroku:

* Visit https://devcenter.heroku.com/articles/getting-started-with-python#set-up
* Install `git`, if you don't have it already.
* Install `heroku`, following instructions for your platform.

Next, we will need to setup our app on Heroku:

* Create a free Heroku account.
* Create a new application.
* Run `heroku login` locally.
* Follow instructions to deploy: https://dashboard.heroku.com/apps/coding-hive-heart-disease/deploy/heroku-git


TODO: Try out this "Deploy to Heroku" button:
- https://devcenter.heroku.com/articles/heroku-button


References:

* Heroku `Procfile` guide: https://devcenter.heroku.com/articles/procfile
* https://xcitech.github.io/tutorials/heroku_tutorial/

# Running on CoCalc

Can easily run the server using Terminal:
https://doc.cocalc.com/howto/webserver.html#raw-files-server

Then access at:
https://cocalc.com/cfca8905-cec9-4b44-b37e-33f41dcc8068/port/5000/

Caveat being that a logged-in session is required

Can use Postman Interceptor to get around this:
https://chrome.google.com/webstore/detail/postman-interceptor/aicmkgpgakddgnaphhhpliifpcfhicfo?hl=en

Turn on interceptor and add filter for your cocal project, e.g.:

    *cocalc.com/cfca8905-cec9-4b44-b37e-33f41dcc8068/port/5000*

In Postman app (instructions here: https://community.getpostman.com/t/interceptor-integration-for-postman-native-apps/5290):
* Click Interceptor icon
* Go to Cookies Tab and
[ABORT!! Requires installing an interceptor app on mac osx ... feels kinda like installing a backdoor so nope..]
