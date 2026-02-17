pipeline {
  agent { label 'myll_ia_bakastov' }

  options {
    timestamps()
    timeout(time: 12, unit: 'MINUTES')
    disableConcurrentBuilds()
  }

  environment {
    VENV  = ".venv"
    PORTS = "8092 8093 8094 8095"
  }

  stages {

    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Prepare venv + deps') {
      steps {
        sh '''#!/usr/bin/env bash
          set -eux
          python3 -V
          rm -rf "$VENV" run_output artifacts.tgz || true

          python3 -m venv "$VENV"
          . "$VENV/bin/activate"

          python -m pip install -U pip
          python -m pip install -r requirements.txt
        '''
      }
    }

    stage('Clean ports (8092-8095)') {
      steps {
        sh '''#!/usr/bin/env bash
          set +e
          for p in $PORTS; do
            fuser -k ${p}/tcp 2>/dev/null || true
          done
          set -e
        '''
      }
    }

    stage('Smoke run (90s, no GUI)') {
      steps {
        sh '''#!/usr/bin/env bash
set -euo pipefail
. "$VENV/bin/activate"

mkdir -p run_output

echo "==> Cleanup old outputs"
rm -f Reciever/*.csv Business/*.csv Business/data_out_*.csv Business/data_metrics_*.csv run_output/*.pid run_output/*.log || true

kill_by_ports() {
  for p in $PORTS; do
    fuser -kv ${p}/tcp >/dev/null 2>&1 || true
  done
}

ports_free() {
  for p in $PORTS; do
    if ss -lnt | grep -q ":${p} "; then
      return 1
    fi
  done
  return 0
}

start_bg() {
  local name="$1"; shift
  setsid nohup "$@" > "run_output/${name}.log" 2>&1 & echo $! > "run_output/${name}.pid"
  echo "Started $name pid=$(cat run_output/${name}.pid)"
}

stop_group() {
  local name="$1"
  if [ -f "run_output/${name}.pid" ]; then
    pid="$(cat run_output/${name}.pid)"
    echo "Stopping $name group (pid=$pid)"
    kill -- "-$pid" >/dev/null 2>&1 || true
  fi
}

wait_ports_ready() {
  for i in $(seq 1 60); do
    ok=0
    for p in $PORTS; do
      if ss -lnt | grep -q ":${p} "; then ok=$((ok+1)); fi
    done
    [ "$ok" -eq 4 ] && return 0
    sleep 0.5
  done
  return 1
}

fail_with_logs() {
  echo "==> FAILURE diagnostics"
  echo "-- listeners:"
  ss -lntp | egrep ':8092|:8093|:8094|:8095' || true
  echo "-- tail logs:"
  tail -n 200 run_output/reciever.log 2>/dev/null || true
  tail -n 200 run_output/simulator.log 2>/dev/null || true
  tail -n 200 run_output/business.log 2>/dev/null || true
}

echo "==> Ensure ports are free"
kill_by_ports
sleep 1
ports_free || (echo "Ports still busy" && ss -lntp | egrep ':8092|:8093|:8094|:8095' && exit 1)

echo "==> Start Reciever FIRST (server)"
start_bg reciever python3 Reciever/reciever.py

echo "==> Wait ports 8092-8095 to become LISTEN"
if ! wait_ports_ready; then
  echo "Ports did not become ready after Reciever start."
  fail_with_logs
  stop_group reciever
  kill_by_ports
  exit 1
fi

echo "==> Start Simulator (client)"
start_bg simulator python3 Simulator/simulator.py

echo "==> Wait until Reciever produces base CSVs (up to 30s)"
need="8092 8093 8094 8095"
for i in $(seq 1 60); do
  ok=0
  for p in $need; do
    f="Reciever/data_port_${p}.csv"
    [ -s "$f" ] && ok=$((ok+1))
  done
  [ "$ok" -eq 4 ] && break
  sleep 0.5
done

for p in $need; do
  f="Reciever/data_port_${p}.csv"
  if [ ! -s "$f" ]; then
    echo "Reciever did not produce $f in time."
    fail_with_logs
    stop_group simulator
    stop_group reciever
    kill_by_ports
    exit 1
  fi
done

echo "==> Start Business (after CSVs exist)"
start_bg business python3 Business/business.py

echo "==> Let them work 90s"
sleep 90

echo "==> Basic checks"
ls -la Reciever/*.csv >/dev/null 2>&1 || (echo "No Reciever CSV" && fail_with_logs && exit 1)

BUS_CSV_COUNT="$(find Business -maxdepth 3 -type f -name '*.csv' 2>/dev/null | wc -l | tr -d ' ')"
echo "Business CSV count: ${BUS_CSV_COUNT}"
[ "${BUS_CSV_COUNT}" -gt 0 ] || (echo "No Business CSV" && fail_with_logs && exit 1)

echo "==> Copy outputs to run_output"
cp -a Reciever/*.csv run_output/ 2>/dev/null || true
find Business -maxdepth 3 -type f -name '*.csv' -exec cp -a {} run_output/ \\; 2>/dev/null || true

echo "==> Stop processes"
stop_group business
stop_group simulator
stop_group reciever

echo "==> Final port cleanup"
kill_by_ports
sleep 1
ports_free || (echo "Ports still busy after stopping" && ss -lntp | egrep ':8092|:8093|:8094|:8095' && exit 1)

echo "Smoke run OK"
'''
      }
    }

    stage('Package artifacts (tgz)') {
      steps {
        sh '''#!/usr/bin/env bash
          set -eux
          tar -czf artifacts.tgz run_output Simulator Reciever Business GUI requirements.txt 2>/dev/null || true
          ls -la artifacts.tgz
        '''
      }
    }
  }

  post {
    always {
      archiveArtifacts artifacts: 'run_output/*, Reciever/*.csv, Business/*.csv, artifacts.tgz', fingerprint: true, allowEmptyArchive: true
      cleanWs()
    }
  }
}