import numpy as np
import json

class SimpleIdentityGallery:
    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.person_counter = 1
        self.gallery = {}  # person_name -> embedding
        self.track_to_person = {}  # track_id -> person_name
        self.person_to_track = {}  # person_name -> track_id (for initial assignment)

    def _generate_person_name(self):
        name = f"person_{self.person_counter:03d}"
        self.person_counter += 1
        return name

    def add_track_embeddings(self, track_id, embeddings, qualities=None):
        """
        Aggregate embeddings for a track and assign a new person name if not already assigned.
        If already assigned, update the gallery embedding for that person.
        """
        if not embeddings:
            return None
        if qualities is not None and len(qualities) == len(embeddings):
            best_idx = int(np.argmax(qualities))
            agg_embedding = embeddings[best_idx]
        else:
            agg_embedding = np.mean(embeddings, axis=0)
        # If this track already has a person, update their embedding
        if track_id in self.track_to_person:
            person_name = self.track_to_person[track_id]
            self.gallery[person_name] = agg_embedding
        else:
            person_name = self._generate_person_name()
            self.gallery[person_name] = agg_embedding
            self.track_to_person[track_id] = person_name
            self.person_to_track[person_name] = track_id
        return person_name

    def assign_identities_for_frame(self, track_embeddings, frame_number):
        """
        Assign person names to tracks in a frame.
        Reuse previous assignment if possible, otherwise match to gallery.
        Ensures no duplicate person_name in the same frame.
        Updates gallery embedding for matched person.
        """
        assigned = {}
        used_persons = set()
        for track_id, emb in track_embeddings.items():
            # Prefer to keep previous assignment if available and not used
            prev_person = self.track_to_person.get(track_id)
            if prev_person and prev_person not in used_persons:
                assigned[track_id] = prev_person
                used_persons.add(prev_person)
                # Optionally update gallery embedding with latest
                self.gallery[prev_person] = emb
                continue
            # Try to match to gallery
            best_person = None
            best_sim = 0.0
            for person_name, gallery_emb in self.gallery.items():
                if person_name in used_persons:
                    continue
                sim = self._cosine_similarity(emb, gallery_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_person = person_name
            if best_person and best_sim >= self.similarity_threshold:
                assigned[track_id] = best_person
                used_persons.add(best_person)
                self.track_to_person[track_id] = best_person
                self.gallery[best_person] = emb  # Update with latest
            else:
                # Assign new person
                new_person = self._generate_person_name()
                self.gallery[new_person] = emb
                assigned[track_id] = new_person
                used_persons.add(new_person)
                self.track_to_person[track_id] = new_person
                self.person_to_track[new_person] = track_id
        return assigned

    @staticmethod
    def _cosine_similarity(a, b):
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def save_gallery(self, filepath):
        """Save gallery (person_name -> embedding) to a JSON file."""
        data = {person: emb.tolist() for person, emb in self.gallery.items()}
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load_gallery(self, filepath):
        """Load gallery (person_name -> embedding) from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.gallery = {person: np.array(emb) for person, emb in data.items()}
        # Reset counters and mappings (no track info)
        self.person_counter = 1 + max([int(p.split('_')[-1]) for p in self.gallery.keys() if p.startswith('person_')], default=0)
        self.track_to_person = {}
        self.person_to_track = {}
