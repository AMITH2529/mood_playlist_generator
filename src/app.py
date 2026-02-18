import os
from flask import Flask, render_template, jsonify, request
from .groq_api import get_artists_from_groq
from .face_mood_analyzer import get_mood_from_webcam
from .spotify_api import create_playlist_from_artists, create_playlist_for_one_artist
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder='../templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_playlist')
def generate_playlist():
    try:
        language = request.args.get('language', None)
        artist = request.args.get('artist', None) 

        mood = get_mood_from_webcam()
        if not mood:
            return jsonify({"success": False, "error": "Could not detect mood."})
        
        print(f"Mood detected: {mood}. Artist requested: {artist}")

        if artist:
            print(f"User requested specific artist: {artist}.")
            playlist_name = f"{artist}'s Top Tracks"
            playlist_url, final_songs = create_playlist_for_one_artist(artist, playlist_name)
            
            if playlist_url:
                 return jsonify({
                    "success": True,
                    "playlist_url": playlist_url,
                    "mood": f"{artist}'s Top Tracks",
                    "songs": final_songs 
                })
            else:
                return jsonify({"success": False, "error": f"Could not find artist: {artist}."})

        else:
            print("No specific artist requested. Getting artists from Groq.")
            
            recommended_artists = get_artists_from_groq(mood, language)
            if not recommended_artists:
                return jsonify({"success": False, "error": "No artists were received from the AI."})

            playlist_name = f"{mood.capitalize()} Mood Playlist"
            playlist_url, final_songs = create_playlist_from_artists(
                artists=recommended_artists, 
                mood=mood,
                playlist_name=playlist_name
            )
        
            if playlist_url:
                return jsonify({
                    "success": True,
                    "playlist_url": playlist_url,
                    "mood": mood.capitalize(),
                    "songs": final_songs 
                })
            else:
                return jsonify({"success": False, "error": "Could not create the Spotify playlist."})

    except Exception as e:
        print(f"An unexpected error occurred in the main app: {e}")
        return jsonify({"success": False, "error": "A server error occurred. Check the terminal for details."})

if __name__ == '__main__':
    app.run(debug=True)