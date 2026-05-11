pipeline {
    agent any

    environment {
        APP_NAME = "emplytics-xai"
    }

    stages {

        stage('Clone Repository') {
            steps {
                git branch: 'main',
                url: 'https://github.com/YOUR_USERNAME/YOUR_REPO.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Run Validation') {
            steps {
                sh 'python -m py_compile dashboard.py'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $APP_NAME .'
            }
        }

        stage('Deployment Ready') {
            steps {
                echo 'Project validated successfully'
            }
        }
    }
}
