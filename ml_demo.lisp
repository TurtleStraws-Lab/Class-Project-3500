;;; ============================================================================
;;; NAME: Vanessa Figueroa
;;; ASGT: Class Project
;;; ORGIN: CSUB-CMPS 3500
;;; FILE: ml_demo.lisp
;;; DATE:12/04/2025
;;; Interactive ML Demo (Common Lisp, SBCL, Linux) — auto-starts on load
;;; ----------------------------------------------------------------------------
;;; • Run:        sbcl --load ml_demo.lisp
;;; • Handles categorical features via one-hot encoding
;;; • Implements:
;;;     - Logistic Regression (binary, L2)
;;;     - Linear Regression (MSE, L2)
;;;     - k-Nearest Neighbors (classification)
;;;     - Decision Tree (ID3-style, Gini, limited depth)
;;;     - Gaussian Naive Bayes (binary)
;;; 
;;; 
;;; ----------------------------------------------------------------------------
;;; Defaults for quick run:
;;;   CSV path default: "data/adult_income_cleaned.csv"
;;;   Target column:    "income"
;;; ============================================================================

#+sbcl (declaim (optimize (speed 1) (safety 3) (debug 3)))

(defpackage :fp
  (:use :cl)
  (:export :main))
(in-package :fp)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Utilities: strings, numbers, RNG
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun trim (s)
  (string-trim '(#\Space #\Tab #\Newline #\Return) s))

(defun parse-number (s)
  "Parse string S to double-float."
  (let ((*read-default-float-format* 'double-float))
    (coerce (read-from-string s) 'double-float)))

(defun parse-float-or-nil (s)
  "Try to parse S to float; NIL if not numeric."
  (handler-case
      (parse-number s)
    (error () nil)))

;; Create a fresh random-state. If SEED is an integer and we're on SBCL,
;; seed it deterministically; otherwise return a fresh state.
(defun make-rng (seed)
  "Create a fresh random-state. If SEED is an integer and SBCL supports seeding,
  use it deterministically; otherwise return a fresh state."
  (cond
    ((and (fboundp 'sb-ext:seed-random-state) (integerp seed))
     (let ((rng (make-random-state t)))
       (sb-ext:seed-random-state rng)
       rng))
    (t
     (make-random-state t))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; UTF-8 CSV I/O
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun read-file-lines (path)
  "Read UTF-8 file PATH, return list of lines."
  (with-open-file (in path :direction :input :external-format :utf-8)
    (loop for line = (read-line in nil nil)
          while line
          collect line)))

(defun split-csv-line (line)
  "Naive CSV split by comma (good for cleaned CSVs)."
  (let ((parts '())
        (start 0)
        (len (length line)))
    (labels ((emit (end)
               (push (subseq line start end) parts)))
      (loop for i from 0 below len do
        (if (char= (char line i) #\,)
            (progn
              (emit i)
              (setf start (1+ i)))
            (when (= i (1- len))
              (emit (1+ i)))))
      (nreverse parts))))

(defun load-csv (path)
  "Return two values: (headers . rows) and row-count."
  (let* ((lines (read-file-lines path)))
    (when (null lines)
      (error "CSV appears empty: ~A" path))
    (let* ((headers (map 'vector #'trim (split-csv-line (first lines))))
           (rows-list (rest lines))
           (rows (make-array (length rows-list))))
      (loop for l in rows-list
            for i from 0 do
              (setf (aref rows i)
                    (map 'vector #'trim (split-csv-line l))))
      (values (cons headers rows)
              (length rows)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Schema detection + one-hot encoding
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defstruct feat-spec
  kind                ;; :num or :cat
  (cats #() :type (vector t))
  (offset 0 :type fixnum)
  (width 1 :type fixnum))

(defun column-index (headers name)
  (or (position name headers :test #'string-equal)
      (error "Column ~A not found in headers: ~A" name headers)))

(defun ensure-binary (y-str)
  "Map common labels to {0,1}; accept numeric strings too."
  (let ((s (string-downcase (trim y-str))))
    (cond
      ((or (string= s "1")
           (string= s ">50k")
           (string= s "> 50k"))
       1.0d0)
      ((or (string= s "0")
           (string= s "<=50k")
           (string= s "<= 50k"))
       0.0d0)
      (t
       (let ((n (parse-float-or-nil s)))
         (if n
             n
             (error "Cannot convert label '~A' to binary" y-str)))))))

(defun detect-schema (headers rows target-name)
  "Return (specs total-width) for non-target columns."
  (let* ((tidx (column-index headers target-name))
         (p (length headers))
         (n (length rows))
         (temp-specs '()))
    (loop for j from 0 below p do
      (unless (= j tidx)
        (let ((all-numeric t)
              (catset (make-hash-table :test #'equal)))
          (loop for i from 0 below n do
            (let ((v (aref (aref rows i) j)))
              (unless (parse-float-or-nil v)
                (setf all-numeric nil)
                (setf (gethash (string-downcase v) catset) t))))
          (if all-numeric
              (push (list :num #() 1) temp-specs)
              (let* ((cats (coerce (loop for k being the hash-keys of catset collect k)
                                   'vector))
                     (k (length cats)))
                (push (list :cat cats (max 1 k)) temp-specs))))))
    (let* ((rev (nreverse temp-specs))
           (specs (make-array (length rev)))
           (offset 0))
      (loop for idx from 0 below (length rev) do
        (destructuring-bind (kind cats width) (nth idx rev)
          (setf (aref specs idx)
                (make-feat-spec :kind kind :cats cats :offset offset :width width))
          (incf offset width)))
      (values specs offset))))


      (defun encode-target-from-rows (rows tidx target-name)
  "Given ROWS (vector of row vectors) and target column index TIDX,
   return a numeric vector Y (double-float).
   - If target values are numeric with >2 unique values → regression (keep numeric).
   - If target values are numeric with exactly 2 unique values → map to 0/1.
   - If target values are strings with exactly 2 unique values → map to 0/1.
   - Otherwise signal an error (multiclass not supported)."
  (let* ((n (length rows))
         (labels (make-array n :element-type 'string)))
    ;; collect raw labels as strings
    (loop for i below n do
      (setf (aref labels i)
            (trim (aref (aref rows i) tidx))))
    ;; try numeric interpretation first
    (let ((numeric-values (make-array n :element-type 'double-float))
          (all-numeric t))
      (loop for i below n do
        (let* ((s (aref labels i))
               (num (parse-float-or-nil s)))
          (if num
              (setf (aref numeric-values i) num)
              (setf all-numeric nil))))
      (cond
        ;; --- all numeric ---
        (all-numeric
         (let* ((unique (remove-duplicates (coerce numeric-values 'list) :test #'=))
                (num-unique (length unique))
                (y (make-array n :element-type 'double-float)))
           (cond
             ;; >2 unique numeric values → regression
             ((> num-unique 2)
              (loop for i below n do
                (setf (aref y i) (aref numeric-values i)))
              y)
             ;; exactly 2 unique numeric values → map to 0/1
             ((= num-unique 2)
              (let* ((sorted (sort (copy-list unique) #'<))
                     (v0 (first sorted))
                     (v1 (second sorted)))
                (loop for i below n do
                  (setf (aref y i)
                        (if (= (aref numeric-values i) v0)
                            0.0d0
                            1.0d0)))
                y))
             ;; all same numeric value → treat as regression constant
             (t
              (loop for i below n do
                (setf (aref y i) (aref numeric-values i)))
              y))))
        ;; --- not all numeric: treat as string labels ---
        (t
         (let* ((unique (remove-duplicates (coerce labels 'list)
                                           :test #'string-equal))
                (num-unique (length unique))
                (y (make-array n :element-type 'double-float)))
           (cond
             ;; exactly 2 distinct string labels → binary classification
             ((= num-unique 2)
              (let* ((u1 (first unique))
                     (u2 (second unique)))
                (loop for i below n do
                  (let ((lab (aref labels i)))
                    (setf (aref y i)
                          (if (string-equal lab u1)
                              0.0d0
                              1.0d0)))))
              y)
             (t
              (error "Target column ~A has ~D distinct categories (~S).~%~
This demo only supports numeric regression or binary (2-class) classification."
                     target-name num-unique unique)))))))))


(defun table->encoded (headers rows target-name)
  "Return (X y specs). X is numeric with one-hot encoding for categoricals.
   Target column is encoded via ENCODE-TARGET-FROM-ROWS."
  (multiple-value-bind (specs totalw)
      (detect-schema headers rows target-name)
    (let* ((tidx (column-index headers target-name))
           (n (length rows))
           (X (make-array (list n totalw) :element-type 'double-float))
           (y (encode-target-from-rows rows tidx target-name)))
      (loop for i from 0 below n do
        (let ((row (aref rows i))
              (feat-col 0))
          ;; build features from all columns except the target column
          (loop for j from 0 below (length headers) do
            (unless (= j tidx)
              (let ((spec (aref specs feat-col)))
                (ecase (feat-spec-kind spec)
                  (:num
                   (let* ((raw (aref row j))
                          (val (parse-float-or-nil raw)))
                     (unless val
                       (error "Non-numeric value in numeric column at row ~D col ~D: ~A"
                              i j raw))
                     (setf (aref X i (feat-spec-offset spec)) val)))
                  (:cat
                   (let* ((raw (string-downcase (aref row j)))
                          (cats (feat-spec-cats spec))
                          (k (feat-spec-width spec))
                          (off (feat-spec-offset spec)))
                     ;; zero slice then set one
                     (loop for c from 0 below k do
                       (setf (aref X i (+ off c)) 0.0d0))
                     (let ((idx (position raw cats :test #'string=)))
                       (when idx
                         (setf (aref X i (+ off idx)) 1.0d0)))))))
              (incf feat-col)))))
      (values X y specs))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Normalization (only numeric columns)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun normalize-numeric-columns (X specs)
  "Return (Xn means stds) with z-score on numeric 1-wide specs; one-hot untouched."
  (let* ((n (array-dimension X 0))
         (d (array-dimension X 1))
         (Xn (make-array (list n d) :element-type 'double-float))
         (means (make-array d :element-type 'double-float :initial-element 0.0d0))
         (stds (make-array d :element-type 'double-float :initial-element 1.0d0)))
    (loop for i below n do
      (loop for j below d do
        (setf (aref Xn i j) (aref X i j))))
    (loop for s across specs do
      (when (and (eq (feat-spec-kind s) :num)
                 (= (feat-spec-width s) 1))
        (let ((j (feat-spec-offset s)))
          (let ((mu 0.0d0))
            (loop for i below n do
              (incf mu (aref Xn i j)))
            (setf mu (/ mu (max 1 n))
                  (aref means j) mu))
          (let ((var 0.0d0))
            (loop for i below n do
              (let ((dval (- (aref Xn i j) (aref means j))))
                (incf var (* dval dval))))
            (let ((sd (max 1d-12 (sqrt (/ var (max 1 (- n 1)))))))
              (setf (aref stds j) sd)
              (loop for i below n do
                (setf (aref Xn i j)
                      (/ (- (aref Xn i j) (aref means j)) sd))))))))
    (values Xn means stds)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Math helpers, split, metrics, timing
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun zeros (n)
  (make-array n :element-type 'double-float :initial-element 0.0d0))

(defun rows (X) (array-dimension X 0))
(defun cols (X) (array-dimension X 1))

(defun add-bias-col (X)
  (let* ((n (rows X))
         (d (cols X))
         (Z (make-array (list n (1+ d)) :element-type 'double-float)))
    (loop for i below n do
      (setf (aref Z i 0) 1.0d0))
    (loop for i below n do
      (loop for j below d do
        (setf (aref Z i (1+ j)) (aref X i j))))
    Z))

(defun mat-vec (X w)
  (let* ((n (rows X))
         (d (cols X))
         (y (zeros n)))
    (loop for i below n do
      (let ((s 0.0d0))
        (loop for j below d do
          (incf s (* (aref X i j) (aref w j))))
        (setf (aref y i) s)))
    y))

(defun sigmoid! (v)
  (loop for i below (length v) do
    (setf (aref v i)
          (/ 1.0d0 (+ 1.0d0 (exp (- (aref v i)))))))
  v)

(defun shuffle-indices (n &optional (seed 42))
  (let ((idx (make-array n :element-type 'fixnum))
        (rng (make-rng seed)))
    (loop for i below n do
      (setf (aref idx i) i))
    (loop for i from (1- n) downto 1 do
      (rotatef (aref idx i)
               (aref idx (random (1+ i) rng))))
    idx))

(defun train-test-split (X y &key (ratio 0.8d0) (seed 42))
  (let* ((n (rows X))
         (ntr (floor (* ratio n)))
         (nte (- n ntr))
         (perm (shuffle-indices n seed)))
    (flet ((takeM (A start count)
             (let* ((d (cols A))
                    (O (make-array (list count d)
                                   :element-type 'double-float)))
               (loop for k below count do
                 (let ((i (aref perm (+ start k))))
                   (loop for j below d do
                     (setf (aref O k j) (aref A i j)))))
               O))
           (takeV (A start count)
             (let ((O (make-array count :element-type 'double-float)))
               (loop for k below count do
                 (setf (aref O k) (aref A (aref perm (+ start k)))))
               O)))
      (values (takeM X 0 ntr)
              (takeV y 0 ntr)
              (takeM X ntr nte)
              (takeV y ntr nte)))))

(defun accuracy (ytrue ypred)
  (let ((n (length ytrue))
        (c 0))
    (loop for i below n do
      (when (= (aref ytrue i) (aref ypred i))
        (incf c)))
    (/ c (max 1 n) 1.0d0)))

(defun confusion-matrix (ytrue ypred)
  "Return 2x2 matrix entries TN, FP, FN, TP as multiple values."
  (let ((tn 0) (fp 0) (fn 0) (tp 0))
    (loop for i below (length ytrue) do
      (let ((yt (aref ytrue i))
            (yp (aref ypred i)))
        (cond
          ((and (= yt 0.0d0) (= yp 0.0d0)) (incf tn))
          ((and (= yt 0.0d0) (= yp 1.0d0)) (incf fp))
          ((and (= yt 1.0d0) (= yp 0.0d0)) (incf fn))
          ((and (= yt 1.0d0) (= yp 1.0d0)) (incf tp)))))
    (values tn fp fn tp)))

(defun f1-for-class (ytrue ypred class-label)
  (let ((tp 0.0d0)
        (fp 0.0d0)
        (fn 0.0d0))
    (loop for i below (length ytrue) do
      (let ((yt (aref ytrue i))
            (yp (aref ypred i)))
        (cond
          ((and (= yt class-label)
                (= yp class-label))
           (incf tp))
          ((and (/= yt class-label)
                (= yp class-label))
           (incf fp))
          ((and (= yt class-label)
                (/= yp class-label))
           (incf fn)))))
    (if (<= (+ (* 2.0d0 tp) fp fn) 0.0d0)
        0.0d0
        (/ (* 2.0d0 tp)
           (+ (* 2.0d0 tp) fp fn)))))

(defun macro-f1 (ytrue ypred)
  (/ (+ (f1-for-class ytrue ypred 0.0d0)
        (f1-for-class ytrue ypred 1.0d0))
     2.0d0))

(defun rmse (ytrue ypred)
  (let* ((n (length ytrue))
         (s 0.0d0))
    (loop for i below n do
      (let ((d (- (aref ypred i) (aref ytrue i))))
        (incf s (* d d))))
    (sqrt (/ s (max 1 n)))))

(defun r2-score (ytrue ypred)
  (let* ((n (length ytrue))
         (mean-y 0.0d0)
         (ss-res 0.0d0)
         (ss-tot 0.0d0))
    (loop for i below n do
      (incf mean-y (aref ytrue i)))
    (setf mean-y (/ mean-y (max 1 n)))
    (loop for i below n do
      (let* ((yi (aref ytrue i))
             (fi (aref ypred i))
             (dy (- yi fi))
             (my (- yi mean-y)))
        (incf ss-res (* dy dy))
        (incf ss-tot (* my my))))
    (if (<= ss-tot 0.0d0)
        0.0d0
        (- 1.0d0 (/ ss-res ss-tot)))))

(defun elapsed-ms (start)
  (* 1000.0d0
     (/ (- (get-internal-real-time) start)
        internal-time-units-per-second)))

        ;;; --------------------------------------------------------------------------
;;; Task type detection + guard for classification algorithms
;;; --------------------------------------------------------------------------

(defun infer-task-type (y)
  "Infer whether label vector Y is regression or binary classification."
  (let* ((values (coerce y 'list))
         (unique (remove-duplicates values :test #'=))
         (num-unique (length unique)))
    (cond
      ;; numeric + exactly 2 unique values → binary classification
      ((and (every #'numberp unique)
            (= num-unique 2))
       :binary-classification)
      ;; numeric + many distinct values → regression
      ((every #'numberp unique)
       :regression)
      ;; fallback: treat as general classification (not supported by 3–6)
      (t
       :classification))))

(defun require-binary-classification (ds)
  "Return T if DS has a binary target, otherwise print a message and return NIL."
  (let* ((y      (getf ds :y))
         (target (getf ds :target))
         (values (coerce y 'list))
         (unique (remove-duplicates values :test #'=))
         (num-unique (length unique)))
    (cond
      ((= num-unique 2)
       t)  ;; OK: exactly two classes
      (t
       (format t "Current target '~A' has ~D unique values (~S).~%~
Classification algorithms (3–6) require a binary target (2 classes, e.g., income).~%"
               target num-unique unique)
       nil)))
)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Logistic Regression (binary, L2)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun randn (n &optional (scale 0.01d0) (seed 7))
  (let ((rng (make-rng seed))
        (v (make-array n :element-type 'double-float)))
    (labels ((gauss ()
               (let* ((u1 (max 1d-12 (- 1 (random 1.0d0 rng))))
                      (u2 (random 1.0d0 rng))
                      (r  (sqrt (* -2.0d0 (log u1))))
                      (th (* 2 pi u2)))
                 (* r (cos th)))))
      (loop for i below n do
        (setf (aref v i) (* scale (gauss)))))
    v))

(defun logistic-train (X y &key (epochs 400) (lr 0.2d0) (l2 0.003d0) (seed 7))
  (let* ((n (rows X))
         (d (cols X))
         (Z (add-bias-col X))
         (w (randn (1+ d) 0.01d0 seed))
         (grad (zeros (1+ d))))
    (dotimes (epoch epochs)
      (declare (ignore epoch))
      (let ((p (sigmoid! (mat-vec Z w))))
        (fill grad 0.0d0)
        ;; grad = (1/n) * Z^T (p - y) + l2 * [0, w1...wd]
        (loop for j below (length w) do
          (let ((s 0.0d0))
            (loop for i below n do
              (incf s (* (aref Z i j)
                         (- (aref p i) (aref y i)))))
            (setf (aref grad j) (/ s n))
            (when (> j 0)
              (incf (aref grad j)
                    (* l2 (aref w j))))))
        ;; gradient step
        (loop for j below (length w) do
          (decf (aref w j) (* lr (aref grad j))))))
    (list :w w)))

(defun logistic-predict (model X)
  (let* ((w (getf model :w))
         (Z (add-bias-col X))
         (p (sigmoid! (mat-vec Z w)))
         (yhat (make-array (length p) :element-type 'double-float)))
    (loop for i below (length p) do
      (setf (aref yhat i)
            (if (>= (aref p i) 0.5d0) 1.0d0 0.0d0)))
    yhat))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Linear Regression (MSE, L2)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun linear-train (X y &key (epochs 300) (lr 0.05d0) (l2 0.0d0) (seed 10))
  (let* ((n (rows X))
         (d (cols X))
         (Z (add-bias-col X))
         (w (randn (1+ d) 0.01d0 seed))
         (grad (zeros (1+ d))))
    (dotimes (epoch epochs)
      (declare (ignore epoch))
      (let ((pred (mat-vec Z w)))
        (fill grad 0.0d0)
        ;; grad = (2/n) * Z^T (pred - y) + 2*l2*[0,w1..wd]
        (loop for j below (length w) do
          (let ((s 0.0d0))
            (loop for i below n do
              (incf s (* (aref Z i j)
                         (- (aref pred i) (aref y i)))))
            (setf (aref grad j) (/ (* 2.0d0 s) n))
            (when (> j 0)
              (incf (aref grad j)
                    (* 2.0d0 l2 (aref w j))))))
        (loop for j below (length w) do
          (decf (aref w j) (* lr (aref grad j))))))
    (list :w w)))

(defun linear-predict (model X)
  (let* ((w (getf model :w))
         (Z (add-bias-col X)))
    (mat-vec Z w)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; k-Nearest Neighbors (classification)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun euclidean-distance (x1 x2)
  (let* ((d (length x1))
         (s 0.0d0))
    (loop for j below d do
      (let ((diff (- (aref x1 j) (aref x2 j))))
        (incf s (* diff diff))))
    (sqrt s)))

(defun knn-train (X y &key (k 7))
  "Training just stores the data and k."
  (list :X X :y y :k k))

(defun knn-predict-one (model x-row)
  (let* ((X (getf model :X))
         (y (getf model :y))
         (k (getf model :k))
         (n (rows X))
         (d (cols X))
         (dist-labels '()))
    (loop for i below n do
      ;; squared distance directly, no temp row
      (let ((s 0.0d0))
        (loop for j below d do
          (let ((diff (- (aref X i j) (aref x-row j))))
            (incf s (* diff diff))))
        (push (cons s (aref y i)) dist-labels)))
    (setf dist-labels (sort dist-labels #'< :key #'car))
    (let ((count0 0)
          (count1 0))
      (loop for pair in dist-labels
            for idx from 0
            while (< idx k) do
              (if (>= (cdr pair) 0.5d0)
                  (incf count1)
                  (incf count0)))
      (if (> count1 count0) 1.0d0 0.0d0))))

(defun knn-predict (model X)
  (let* ((n (rows X))
         (d (cols X))
         (tmp (make-array d :element-type 'double-float))
         (yhat (make-array n :element-type 'double-float)))
    (loop for i below n do
      ;; copy row i into tmp
      (loop for j below d do
        (setf (aref tmp j) (aref X i j)))
      (setf (aref yhat i)
            (knn-predict-one model tmp)))
    yhat))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Decision Tree (ID3-style, Gini) for binary classification
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defstruct dt-node
  leaf?
  label      ; double-float for leaf
  feat       ; feature index
  thr        ; threshold
  left
  right)

(defun majority-label (y idxs)
  "Return 1.0d0 if ones >= zeros, else 0.0d0, for indices in IDXS."
  (let ((ones 0)
        (zeros 0))
    (loop for i below (length idxs) do
      (if (>= (aref y (aref idxs i)) 0.5d0)
          (incf ones)
          (incf zeros)))
    (if (>= ones zeros) 1.0d0 0.0d0)))

(defun gini2 (p n)
  (let ((tot (+ p n)))
    (if (= tot 0)
        0.0d0
        (let ((p1 (/ p tot))
              (p0 (/ n tot)))
          (- 1.0d0 (+ (* p1 p1) (* p0 p0)))))))

(defun dt-build (X y idxs depth max-depth min-samples)
  "Recursive helper that builds a decision tree node."
  (let* ((n (length idxs))
         (ones 0)
         (zeros 0))
    ;; count class labels in this node
    (loop for i below n do
      (if (>= (aref y (aref idxs i)) 0.5d0)
          (incf ones)
          (incf zeros)))
    ;; stopping criteria -> leaf
    (when (or (= ones 0)
              (= zeros 0)
              (>= depth max-depth)
              (<= n min-samples))
      (return-from dt-build
        (make-dt-node :leaf? t
                      :label (if (>= ones zeros) 1.0d0 0.0d0)
                      :feat -1
                      :thr 0.0d0
                      :left nil
                      :right nil)))
    ;; search best split
    (let* ((d (cols X))
           (best-gini 1.0d0)
           (best-feat -1)
           (best-thr 0.0d0))
      (loop for j below d do
        (let ((all-binary t)
              (sum 0.0d0))
          ;; detect if feature is 0/1 only and compute mean
          (loop for i below n do
            (let* ((row (aref idxs i))
                   (v (aref X row j)))
              (incf sum v)
              (unless (or (= v 0.0d0) (= v 1.0d0))
                (setf all-binary nil))))
          (let* ((thr (if all-binary
                          0.5d0
                          (/ sum (max 1.0d0 n))))
                 (l-pos 0) (l-neg 0)
                 (r-pos 0) (r-neg 0))
            (loop for i below n do
              (let* ((row (aref idxs i))
                     (v (aref X row j))
                     (lab (if (>= (aref y row) 0.5d0) 1 0)))
                (if (<= v thr)
                    (if (= lab 1)
                        (incf l-pos)
                        (incf l-neg))
                    (if (= lab 1)
                        (incf r-pos)
                        (incf r-neg)))))
            (let* ((l-n (+ l-pos l-neg))
                   (r-n (+ r-pos r-neg)))
              (when (and (> l-n 0) (> r-n 0))
                (let* ((g-left  (gini2 l-pos l-neg))
                       (g-right (gini2 r-pos r-neg))
                       (g (/ (+ (* g-left  l-n)
                                (* g-right r-n))
                             (+ l-n r-n))))
                  (when (< g best-gini)
                    (setf best-gini g
                          best-feat j
                          best-thr thr)))))))
      ;; if we never improved, make a leaf with majority label
      (when (= best-feat -1)
        (return-from dt-build
          (make-dt-node :leaf? t
                        :label (if (>= ones zeros) 1.0d0 0.0d0)
                        :feat -1
                        :thr 0.0d0
                        :left nil
                        :right nil)))
      ;; otherwise split indices and recurse
      (let* ((left-count 0)
             (right-count 0))
        (loop for i below n do
          (let* ((row (aref idxs i))
                 (v (aref X row best-feat)))
            (if (<= v best-thr)
                (incf left-count)
                (incf right-count))))
        (let ((left-idxs  (make-array left-count  :element-type 'fixnum))
              (right-idxs (make-array right-count :element-type 'fixnum))
              (li 0)
              (ri 0))
          (loop for i below n do
            (let* ((row (aref idxs i))
                   (v (aref X row best-feat)))
              (if (<= v best-thr)
                  (progn
                    (setf (aref left-idxs li) row)
                    (incf li))
                  (progn
                    (setf (aref right-idxs ri) row)
                    (incf ri)))))
          (make-dt-node
           :leaf? nil
           :label 0.0d0
           :feat best-feat
           :thr best-thr
           :left  (dt-build X y left-idxs  (1+ depth) max-depth min-samples)
           :right (dt-build X y right-idxs (1+ depth) max-depth min-samples))))))))


(defun id3-train (X y &key (max-depth 5) (min-samples 200))
  "Entry point: prepare index vector, then call dt-build."
  (let* ((n (rows X))
         (idxs (make-array n :element-type 'fixnum)))
    (loop for i below n do
      (setf (aref idxs i) i))
    (dt-build X y idxs 0 max-depth min-samples)))

(defun id3-predict-one (node x-row)
  (if (dt-node-leaf? node)
      (dt-node-label node)
      (if (<= (aref x-row (dt-node-feat node))
              (dt-node-thr node))
          (id3-predict-one (dt-node-left node) x-row)
          (id3-predict-one (dt-node-right node) x-row))))

(defun id3-predict (model X)
  (let* ((n (rows X))
         (d (cols X))
         (yhat (make-array n :element-type 'double-float)))
    (loop for i below n do
      (let ((row (make-array d :element-type 'double-float)))
        (loop for j below d do
          (setf (aref row j) (aref X i j)))
        (setf (aref yhat i)
              (if (>= (id3-predict-one model row) 0.5d0)
                  1.0d0
                  0.0d0))))
    yhat))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Gaussian Naive Bayes (binary classification)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun gnb-train (X y)
  "Fit Gaussian Naive Bayes for labels {0,1}."
  (let* ((n (rows X))
         (d (cols X))
         (means (make-array (list 2 d)
                            :element-type 'double-float
                            :initial-element 0.0d0))
         (vars  (make-array (list 2 d)
                            :element-type 'double-float
                            :initial-element 0.0d0))
         (counts (make-array 2
                             :element-type 'double-float
                             :initial-element 0.0d0)))
    ;; means
    (loop for i below n do
      (let* ((cls (if (>= (aref y i) 0.5d0) 1 0)))
        (incf (aref counts cls))
        (loop for j below d do
          (incf (aref means cls j) (aref X i j)))))
    (loop for c from 0 to 1 do
      (when (> (aref counts c) 0)
        (loop for j below d do
          (setf (aref means c j)
                (/ (aref means c j) (aref counts c))))))
    ;; variances
    (loop for i below n do
      (let* ((cls (if (>= (aref y i) 0.5d0) 1 0)))
        (loop for j below d do
          (let* ((xij (aref X i j))
                 (mc (aref means cls j))
                 (diff (- xij mc)))
            (incf (aref vars cls j) (* diff diff))))))
    (loop for c from 0 to 1 do
      (if (> (aref counts c) 1)
          (loop for j below d do
            (setf (aref vars c j)
                  (max 1d-6
                       (/ (aref vars c j)
                          (1- (aref counts c))))))
          (loop for j below d do
            (setf (aref vars c j) 1.0d0))))
    (let ((priors (make-array 2 :element-type 'double-float)))
      (loop for c from 0 to 1 do
        (setf (aref priors c)
              (/ (aref counts c) (max 1.0d0 n))))
      (list :means means :vars vars :priors priors))))

(defun gnb-loglik (x means vars prior cls)
  "Log p(x | class=CLS) + log prior, using 2D MEANS/VARS arrays."
  (let* ((d (length x))
         (s (log prior)))
    (loop for j below d do
      (let* ((mu  (aref means cls j))
             (var (aref vars  cls j))
             (xj  (aref x j)))
        (incf s
              (- (/ (expt (- xj mu) 2)
                    (* 2.0d0 var))
                 (* 0.5d0 (log (* 2.0d0 pi var)))))))
    s))

(defun gnb-predict (model X)
  (let* ((means  (getf model :means))
         (vars   (getf model :vars))
         (priors (getf model :priors))
         (n (rows X))
         (d (cols X))
         (yhat (make-array n :element-type 'double-float)))
    (loop for i below n do
      (let ((row (make-array d :element-type 'double-float)))
        ;; copy row i
        (loop for j below d do
          (setf (aref row j) (aref X i j)))
        ;; log-likelihoods for class 0 and 1
        (let* ((ll0 (gnb-loglik row means vars (aref priors 0) 0))
               (ll1 (gnb-loglik row means vars (aref priors 1) 1)))
          (setf (aref yhat i)
                (if (> ll1 ll0) 1.0d0 0.0d0)))))
    yhat))


(defun gnb-run (ds)
  (multiple-value-bind (Xtr ytr Xte yte)
      (train-test-split (getf ds :X) (getf ds :y) :seed 21)
    (let ((start (get-internal-real-time)))
      (let* ((model (gnb-train Xtr ytr))
             (yhat  (gnb-predict model Xte))
             (acc   (accuracy yte yhat))
             (f1    (macro-f1 yte yhat))
             (ms    (elapsed-ms start)))
        (multiple-value-bind (tn fp fn tp)
            (confusion-matrix yte yhat)
          (format t "Algorithm: Gaussian Naive Bayes~%")
          (format t "Accuracy: ~,4F   Macro-F1: ~,4F~%" acc f1)
          (format t "Confusion matrix [tn fp; fn tp] = [~D ~D; ~D ~D]~%"
                  tn fp fn tp)
          (format t "Time: ~,2F ms~%" ms))))))

          ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; "Runner" helpers for each algorithm (metrics + timing)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun linear-regression-run (ds)
  (multiple-value-bind (Xtr ytr Xte yte)
      (train-test-split (getf ds :X) (getf ds :y) :seed 123)
    (let ((start (get-internal-real-time)))
      (let* ((model (linear-train Xtr ytr :epochs 300 :lr 0.05d0))
             (yhat  (linear-predict model Xte))
             (error (rmse yte yhat))
             (r2    (r2-score yte yhat))
             (ms    (elapsed-ms start)))
        (format t "Algorithm: Linear Regression~%")
        (format t "RMSE: ~,4F   R^2: ~,4F~%" error r2)
        (format t "Time: ~,2F ms~%" ms)))))

(defun logistic-regression-run (ds &key (epochs 400) (lr 0.2d0) (l2 0.003d0))
  (multiple-value-bind (Xtr ytr Xte yte)
      (train-test-split (getf ds :X) (getf ds :y) :seed 42)
    (let ((start (get-internal-real-time)))
      (let* ((model (logistic-train Xtr ytr
                                    :epochs epochs
                                    :lr lr
                                    :l2 l2
                                    :seed 7))
             (yhat (logistic-predict model Xte))
             (acc  (accuracy yte yhat))
             (f1   (macro-f1 yte yhat))
             (ms   (elapsed-ms start)))
        (multiple-value-bind (tn fp fn tp)
            (confusion-matrix yte yhat)
          (format t "Algorithm: Logistic Regression~%")
          (format t "Accuracy: ~,4F   Macro-F1: ~,4F~%" acc f1)
          (format t "Confusion matrix [tn fp; fn tp] = [~D ~D; ~D ~D]~%"
                  tn fp fn tp)
          (format t "Time: ~,2F ms~%" ms))))))

(defun knn-run (ds &key (k 7) (max-train 4000) (max-test 4000))
  (multiple-value-bind (Xtr ytr Xte yte)
      (train-test-split (getf ds :X) (getf ds :y) :seed 99)
    (let* ((ntr (min (rows Xtr) max-train))
           (nte (min (rows Xte) max-test))
           (d   (cols Xtr))
           (Xtr-small (make-array (list ntr d)
                                  :element-type 'double-float))
           (ytr-small (make-array ntr :element-type 'double-float))
           (Xte-small (make-array (list nte d)
                                  :element-type 'double-float))
           (yte-small (make-array nte :element-type 'double-float)))
      ;; copy subsets
      (loop for i below ntr do
        (loop for j below d do
          (setf (aref Xtr-small i j) (aref Xtr i j)))
        (setf (aref ytr-small i) (aref ytr i)))
      (loop for i below nte do
        (loop for j below d do
          (setf (aref Xte-small i j) (aref Xte i j)))
        (setf (aref yte-small i) (aref yte i)))

      (let ((start (get-internal-real-time)))
        (let* ((model (knn-train Xtr-small ytr-small :k k))
               (yhat  (knn-predict model Xte-small))
               (acc   (accuracy yte-small yhat))
               (f1    (macro-f1 yte-small yhat))
               (ms    (elapsed-ms start)))
          (multiple-value-bind (tn fp fn tp)
              (confusion-matrix yte-small yhat)
            (format t "Algorithm: k-Nearest Neighbors (k=~D, train ~D, test ~D)~%"
                    k ntr nte)
            (format t "Accuracy: ~,4F   Macro-F1: ~,4F~%" acc f1)
            (format t "Confusion matrix [tn fp; fn tp] = [~D ~D; ~D ~D]~%"
                    tn fp fn tp)
            (format t "Time: ~,2F ms~%" ms)))))))

(defun id3-run (ds &key (max-depth 3)
                        (min-samples 1200)
                        (max-train 1000)
                        (max-test 2000))
  "Run ID3 on a subsampled train/test split for speed."
  (multiple-value-bind (Xtr ytr Xte yte)
      (train-test-split (getf ds :X) (getf ds :y) :seed 5)
    ;; --- subsample training and test sets for speed ---
    (let* ((ntr  (min (rows Xtr) max-train))
           (nte  (min (rows Xte) max-test))
           (d    (cols Xtr))
           (Xtr-small (make-array (list ntr d)
                                  :element-type 'double-float))
           (ytr-small (make-array ntr :element-type 'double-float))
           (Xte-small (make-array (list nte d)
                                  :element-type 'double-float))
           (yte-small (make-array nte :element-type 'double-float)))
      ;; copy subsets
      (loop for i below ntr do
        (loop for j below d do
          (setf (aref Xtr-small i j) (aref Xtr i j)))
        (setf (aref ytr-small i) (aref ytr i)))
      (loop for i below nte do
        (loop for j below d do
          (setf (aref Xte-small i j) (aref Xte i j)))
        (setf (aref yte-small i) (aref yte i)))

      (format t "Training ID3 on ~D rows (max-depth=~D, min-samples=~D)...~%"
              ntr max-depth min-samples)
      (finish-output)

      ;; --- train on the smaller subset ---
      (let ((start (get-internal-real-time)))
        (let* ((model (id3-train Xtr-small ytr-small
                                 :max-depth max-depth
                                 :min-samples min-samples))
               (yhat (id3-predict model Xte-small))
               (acc  (accuracy yte-small yhat))
               (f1   (macro-f1 yte-small yhat))
               (ms   (elapsed-ms start)))
          (multiple-value-bind (tn fp fn tp)
              (confusion-matrix yte-small yhat)
            (format t "Algorithm: Decision Tree (ID3, train ~D, test ~D)~%"
                    ntr nte)
            (format t "Accuracy: ~,4F   Macro-F1: ~,4F~%" acc f1)
            (format t "Confusion matrix [tn fp; fn tp] = [~D ~D; ~D ~D]~%"
                    tn fp fn tp)
            (format t "Time: ~,2F ms~%" ms)))))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Interactive Menu (auto-starts on load)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun prompt (msg)
  (format t "~A" msg)
  (finish-output)
  (read-line))

(defun print-menu ()
  (format t "~%=== ML (Common Lisp) Demo ===~%")
  (format t "(1) Load dataset~%")
  (format t "(2) Linear Regression (regression)~%")
  (format t "(3) Logistic Regression (binary)~%")
  (format t "(4) k-Nearest Neighbors (classification)~%")
  (format t "(5) Decision Tree ID3 (classification)~%")
  (format t "(6) Gaussian Naive Bayes (classification)~%")
  (format t "(7) Quit~%"))

(defun main ()
  "Interactive menu loop. Auto-invoked at end of file."
  (let ((ds nil)
        (specs nil))
    (loop
      (print-menu)
      (let ((choice (trim (prompt "Enter option: "))))
        (cond
          ;; Load dataset
          ((string= choice "1")
           (let* ((path (trim (prompt
                               "CSV path [default: data/adult_income_cleaned.csv]: ")))
                  (path (if (string= path "")
                            "data/adult_income_cleaned.csv"
                            path))
                  (target (trim (prompt
                                 "Target column [default: income]: ")))
                  (target (if (string= target "")
                              "income"
                              target))
                  (norm? (string= (string-downcase
                                   (trim (prompt
                                          "Normalize numeric features? (y/N): ")))
                                  "y")))
             (handler-case
                 (multiple-value-bind (table _rows)
                     (load-csv path)
                   (declare (ignore _rows))
                   (destructuring-bind (headers . rows) table
                     (handler-case
                         (multiple-value-bind (X y sp)
                             (table->encoded headers rows target)
                           (setf specs sp)
                           (when norm?
                             (multiple-value-bind (Xn means stds)
                                 (normalize-numeric-columns X specs)
                               (declare (ignore means stds))
                               (setf X Xn)))
                           ;; store dataset + metadata (target name + task type)
                           (let ((task-type (infer-task-type y)))
                             (setf ds (list :X X :y y
                                            :target target
                                            :task-type task-type))
                             (format t "Loaded ~D rows, ~D features (after encoding).~%"
                                     (rows X) (cols X))
                             (format t "Target: ~A (~A)~%" target task-type)))
                       (error (e)
                         (format t "Error during encoding: ~A~%" e)))))
               (file-error (e)
                 (format t "Could not open file: ~A~%" e))
               (error (e)
                 (format t "Error while loading CSV: ~A~%" e)))))

          ;; Linear Regression
          ((string= choice "2")
           (if (null ds)
               (format t "Load data first (option 1).~%")
               (linear-regression-run ds)))

          ;; Logistic Regression with runtime params (binary only)
          ((string= choice "3")
           (if (null ds)
               (format t "Load data first (option 1).~%")
               (when (require-binary-classification ds)
                 (let* ((epochs-str (trim (prompt "Epochs [default: 400]: ")))
                        (lr-str     (trim (prompt "Learning rate [default: 0.2]: ")))
                        (l2-str     (trim (prompt "L2 [default: 0.003]: ")))
                        (epochs (if (string= epochs-str "")
                                    400
                                    (parse-integer epochs-str)))
                        (lr     (if (string= lr-str "")
                                    0.2d0
                                    (parse-number lr-str)))
                        (l2     (if (string= l2-str "")
                                    0.003d0
                                    (parse-number l2-str))))
                   (logistic-regression-run ds
                                            :epochs epochs
                                            :lr lr
                                            :l2 l2)))))

          ;; k-NN (binary only)
          ((string= choice "4")
           (if (null ds)
               (format t "Load data first (option 1).~%")
               (when (require-binary-classification ds)
                 (let* ((k-str (trim (prompt "k [default: 7]: ")))
                        (k (if (string= k-str "")
                               7
                               (parse-integer k-str))))
                   (knn-run ds :k k)))))

          ;; ID3 Decision Tree (binary only)
          ((string= choice "5")
           (if (null ds)
               (format t "Load data first (option 1).~%")
               (when (require-binary-classification ds)
                 ;; call with no extra kwargs → uses id3-run's defaults
                 (id3-run ds))))

          ;; Gaussian Naive Bayes (binary only)
          ((string= choice "6")
           (if (null ds)
               (format t "Load data first (option 1).~%")
               (when (require-binary-classification ds)
                 (gnb-run ds))))

          ;; Quit
          ((string= choice "7")
           (return (format t "Goodbye!~%")))

          (t
           (format t "Invalid option.~%")))))))

;;; Auto-start menu upon load:
(fp:main)


;;; End of file
