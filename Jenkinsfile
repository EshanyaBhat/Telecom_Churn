pipeline {
    agent any
    stages {
        stage('Build image') {
            steps {
                sh 'docker build -t my-image .'
            }
        }
        stage('Test image') {
            steps {
                sh 'docker run my-image pytest'
            }
        }
        stage('Deploy image') {
            steps {
                sh 'docker push my-image'
                sh 'ssh user@server "docker pull my-image && docker run -d --name my-app -p 8080:8080 my-image"'
            }
        }
    }
}
