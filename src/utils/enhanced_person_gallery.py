# This file has been removed as part of switching to FAISS-only gallery system
# The enhanced JSON-based gallery has been replaced with FAISSPersonGallery
# Please use src.utils.faiss_gallery.FAISSPersonGallery instead

# Removed classes:
# - MovementType, OrientationType (enums)
# - MovementProfile, PersonEmbedding, EnhancedPersonData (dataclasses) 
# - MovementOrientationAnalyzer, EnhancedPersonGallery (main classes)

# Migration guide:
# - Replace EnhancedPersonGallery with FAISSPersonGallery
# - Use faiss_gallery.add_person_embedding() instead of enhanced_gallery.add_person_embedding()
# - Use faiss_gallery.identify_person() instead of enhanced_gallery.identify_person()
# - Use faiss_gallery.get_gallery_statistics() instead of enhanced_gallery.print_gallery_report()
