## Postgres setup of DB
You may find it helpful to create the database's user and schema in PostgreSQL:

Log into PostgreSQL from the postgres user
$ sudo -u postgres psql postgres

Once in, create the user and database

-CREATE ROLE myuser LOGIN PASSWORD 'mypass';
-CREATE DATABASE mydatabase WITH OWNER = myuser;

Log into PostgreSQL from the new user account
$ psql -h localhost -d mydatabase -U myuser -p <port>


pg-admin:
sudo /usr/pgadmin4/bin/setup-web.sh

pwd: holdon
weblink for pgadmin: http://127.0.0.1/pgadmin4


## Mp4 to wav
ffmpeg -i ../video/youtube_video/first_person_vid.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 first_person_audio.wav
