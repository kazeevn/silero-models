# The LiteRT JNI layer creates and calls these classes by name
-keep class com.google.ai.edge.litert.** { *; }
-keep class org.tensorflow.lite.** { *; }

# LiteRT pulls in WorkManager, whose initializer opens a Room database:
# Room instantiates the generated WorkDatabase_Impl reflectively, by name.
# R8 does not keep it instantiable, and the app crashes on start
# ("Failed to create an instance of class androidx.work.impl.WorkDatabase").
-keep class * extends androidx.room.RoomDatabase { <init>(); }
