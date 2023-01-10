pipeline {
  options {
    disableConcurrentBuilds()
  }

  agent {
    label "nodejs-app-slave"
  }

  parameters {
    string(name: 'COMMIT_ID', defaultValue: '', description: "The git commit id to indicate the codebase for building. By default it's the environment GIT_COMMIT from Jenkins.")
    booleanParam(name: 'BUILD', defaultValue: true, description: "Build and publish Docker image to AWS ECR.")
    booleanParam(name: 'DEPLOY_TO_KOBITON_TEST', defaultValue: false, description: 'Deploy to Kobiton Test environment')
    booleanParam(name: 'DEPLOY_TO_KOBITON_STAGING', defaultValue: false, description: 'Deploy to Kobiton Staging environment')
    booleanParam(name: "DEPLOY_TO_KOBITON_PRODUCTION", defaultValue: false, description: "Deploy an existing build to Kobiton Production environment")
    booleanParam(name: "APPLY_K8S_VIRTUAL_SERVICE_MANIFEST", defaultValue: false, description: "Apply K8s Virtual Service manifest, which will affect traffic switching")
    string(name: 'DEPLOY_REPO_CODEBASE_ID', defaultValue: 'origin/master', description: 'Either the remote branch ("origin/<branch name>"),  git commit id ("aaec22") or git tag (release-v3.6.0) of the env repo that is used to deploy the app. By default it is "origin/master"')
  }

  environment {
    REPO_NAME = "kobiton/ita-ai-service"
    CI_DEPLOY_AI_RUNNER = "ai-service-runner"
    CI_DEPLOY_AI_BGR = "ai-service-background-service"
    CI_DOCKER_IMAGE_NAME = "ai-service"
  }

  stages {
    stage('Set variables for CI') {
      steps {
        script {
          GIT_COMMIT_ID=sh(script: "[ -z $params.COMMIT_ID ] && echo ${env.GIT_COMMIT.take(6)} || git rev-parse --short=6 $params.COMMIT_ID", returnStdout: true).trim()
        }
      }
    }

    stage('Pull docker-ci tool') {
      when {
        expression {
          return !env.BRANCH_NAME.startsWith("PR-")
        }
      }

      steps {
        script {
          // login to AWS ECR
          sh('eval $(aws ecr get-login --no-include-email --region ap-southeast-1)')

          sh("docker pull 580359070243.dkr.ecr.ap-southeast-1.amazonaws.com/docker-ci:latest")

          // Alias it to call below statement
          sh("docker tag 580359070243.dkr.ecr.ap-southeast-1.amazonaws.com/docker-ci:latest docker-ci:latest")
        }
      }
    }

    stage('Build and publish to AWS ECR') {
      when {
        expression {
          return !env.BRANCH_NAME.startsWith("PR-") && params.BUILD
        }
      }

      steps {
        script {
          sh("docker run --rm \
                -v /var/run/docker.sock:/var/run/docker.sock \
                docker-ci:latest \
                  --git-url $env.GIT_URL \
                  --commit-id $GIT_COMMIT_ID \
                  build --docker --docker-image-name $env.CI_DOCKER_IMAGE_NAME \
                  --docker-image-tag $GIT_COMMIT_ID")
          sh("docker run --rm \
                -v /var/run/docker.sock:/var/run/docker.sock \
                docker-ci:latest archive --docker-registry \
                --docker-image-name $env.CI_DOCKER_IMAGE_NAME \
                --docker-image-tag $GIT_COMMIT_ID")
        }
      }
    }

    stage("Deploy to Kobiton Test") {
      when {
        expression {
          return !env.BRANCH_NAME.startsWith("PR-") && params.DEPLOY_TO_KOBITON_TEST
        }
      }

      steps {
        script {
          sh("docker run --rm \
            -v /var/run/docker.sock:/var/run/docker.sock \
            docker-ci:latest deploy \
            --env-name kobiton-test --app-name $env.CI_DEPLOY_AI_RUNNER \
            ${params.APPLY_K8S_VIRTUAL_SERVICE_MANIFEST? "--extra-k8s-apply-virtual-service" : ''} \
            --env-codebase-id $params.DEPLOY_REPO_CODEBASE_ID \
            --docker-image-tag $GIT_COMMIT_ID")
          sh("docker run --rm \
            -v /var/run/docker.sock:/var/run/docker.sock \
            docker-ci:latest deploy \
            --env-name kobiton-test --app-name $env.CI_DEPLOY_AI_BGR \
            ${params.APPLY_K8S_VIRTUAL_SERVICE_MANIFEST? "--extra-k8s-apply-virtual-service" : ''} \
            ${params.DEPLOY_REPO_CODEBASE_ID ? "--env-codebase-id $params.DEPLOY_REPO_CODEBASE_ID" : ''} \
            --env-codebase-id $params.DEPLOY_REPO_CODEBASE_ID \
            --docker-image-tag $GIT_COMMIT_ID")
        }
      }
    }

    stage("Deploy to Kobiton Staging") {
      when {
        expression {
          return params.DEPLOY_TO_KOBITON_STAGING
        }
      }

      steps {
        script {
          sh("docker run --rm \
            -v /var/run/docker.sock:/var/run/docker.sock \
            docker-ci:latest deploy \
            --env-name kobiton-staging --app-name $env.CI_DEPLOY_AI_RUNNER \
            ${params.APPLY_K8S_VIRTUAL_SERVICE_MANIFEST? "--extra-k8s-apply-virtual-service" : ''} \
            --env-codebase-id $params.DEPLOY_REPO_CODEBASE_ID \
            --docker-image-tag $GIT_COMMIT_ID")
          sh("docker run --rm \
            -v /var/run/docker.sock:/var/run/docker.sock \
            docker-ci:latest deploy \
            --env-name kobiton-staging --app-name $env.CI_DEPLOY_AI_BGR \
            ${params.APPLY_K8S_VIRTUAL_SERVICE_MANIFEST? "--extra-k8s-apply-virtual-service" : ''} \
            ${params.DEPLOY_REPO_CODEBASE_ID ? "--env-codebase-id $params.DEPLOY_REPO_CODEBASE_ID" : ''} \
            --env-codebase-id $params.DEPLOY_REPO_CODEBASE_ID \
            --docker-image-tag $GIT_COMMIT_ID")
        }
      }
    }

    stage("Deploy to Kobiton Production") {
      when {
        expression {
          return params.DEPLOY_TO_KOBITON_PRODUCTION
        }
      }

      steps {
        script {
          sh("docker run --rm \
            -v /var/run/docker.sock:/var/run/docker.sock \
            docker-ci:latest deploy \
            --env-name kobiton-prod --app-name $env.CI_DEPLOY_AI_RUNNER \
            ${params.APPLY_K8S_VIRTUAL_SERVICE_MANIFEST? "--extra-k8s-apply-virtual-service" : ''} \
            --env-codebase-id $params.DEPLOY_REPO_CODEBASE_ID \
            --docker-image-tag $GIT_COMMIT_ID")
          sh("docker run --rm \
            -v /var/run/docker.sock:/var/run/docker.sock \
            docker-ci:latest deploy \
            --env-name kobiton-prod --app-name $env.CI_DEPLOY_AI_BGR \
            ${params.APPLY_K8S_VIRTUAL_SERVICE_MANIFEST? "--extra-k8s-apply-virtual-service" : ''} \
            ${params.DEPLOY_REPO_CODEBASE_ID ? "--env-codebase-id $params.DEPLOY_REPO_CODEBASE_ID" : ''} \
            --env-codebase-id $params.DEPLOY_REPO_CODEBASE_ID \
            --docker-image-tag $GIT_COMMIT_ID")
        }
      }
    }
  }

  post {
    failure {
      script {
        sh("docker run --rm \
              docker-ci:latest notify \
              --failure \
              --job-param-name COMMIT_ID --job-param-value ${GIT_COMMIT_ID} \
              --job-param-name BUILD --job-param-value ${params.BUILD} \
              --job-param-name DEPLOY_TO_KOBITON_TEST --job-param-value ${params.DEPLOY_TO_KOBITON_TEST} \
              --job-param-name DEPLOY_TO_KOBITON_STAGING --job-param-value ${params.DEPLOY_TO_KOBITON_STAGING} \
              --job-param-name DEPLOY_TO_KOBITON_PRODUCTION --job-param-value ${params.DEPLOY_TO_KOBITON_PRODUCTION} \
              ${params.APPLY_K8S_VIRTUAL_SERVICE_MANIFEST ? "--job-param-name APPLY_K8S_VIRTUAL_SERVICE_MANIFEST --job-param-value $params.APPLY_K8S_VIRTUAL_SERVICE_MANIFEST": ''} \
              --ci-job-url ${env.BUILD_URL} \
              --ci-job-id ${env.BUILD_ID} \
              --git-repo-name $env.REPO_NAME")
      }
    }

    success {
      script {
        sh("docker run --rm \
              docker-ci:latest notify \
              --job-param-name COMMIT_ID --job-param-value ${GIT_COMMIT_ID} \
              --job-param-name BUILD --job-param-value ${params.BUILD} \
              --job-param-name DEPLOY_TO_KOBITON_TEST --job-param-value ${params.DEPLOY_TO_KOBITON_TEST} \
              --job-param-name DEPLOY_TO_KOBITON_STAGING --job-param-value ${params.DEPLOY_TO_KOBITON_STAGING} \
              --job-param-name DEPLOY_TO_KOBITON_PRODUCTION --job-param-value ${params.DEPLOY_TO_KOBITON_PRODUCTION} \
              ${params.APPLY_K8S_VIRTUAL_SERVICE_MANIFEST ? "--job-param-name APPLY_K8S_VIRTUAL_SERVICE_MANIFEST --job-param-value $params.APPLY_K8S_VIRTUAL_SERVICE_MANIFEST": ''} \
              --ci-job-url ${env.BUILD_URL} \
              --ci-job-id ${env.BUILD_ID} \
              --git-repo-name $env.REPO_NAME")
      }
    }
  }
}
