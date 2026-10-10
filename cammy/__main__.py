from .main import main

# The guard matters: a child process (the AI model runs in one) re-imports the entry module,
# and without it that import would start a second copy of the app.
if __name__ == "__main__":
    main()
